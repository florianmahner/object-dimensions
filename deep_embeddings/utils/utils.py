#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Miscellaneous utility classes and functions."""


import argparse
import torch
import os
import math
import glob
import scipy.stats
import torchvision
import toml
import sys

import torch.nn.functional as F
import numpy as np

from numba import njit, prange
from scipy.spatial.distance import pdist, squareform
from deep_embeddings.utils.image_dataset import ImageDataset


class ExperimentParser:
    """Convenience argument parser for all experiments also accepts a .toml file as input"""

    def __init__(self, *args, **kwargs):
        self.parser = argparse.ArgumentParser(*args, **kwargs)
        self.parser.add_argument(
            "--config",
            type=str,
            default="",
            help="Path to the configuration file.",
        )
        self.parser.add_argument(
            "--section", 
            type=str, 
            default="",
            help="Section name in the config file to parse arguments from")

    def add_argument(self, *args, **kwargs):
        return self.parser.add_argument(*args, **kwargs)

    def _extract_args(self):
        """ Find the default arguments of the argument parser if any and the ones that are passed
        through the command line"""
        sys_defaults = sys.argv.copy()
        sys.argv = []
        default_args = self.parser.parse_args()
        sys.argv = sys_defaults
        cmdl_args = self.parser.parse_args()
        return default_args, cmdl_args

    def _find_changed_args(self, default_args, cmdl_args):
        """Find the arguments that have been changed from the command line to replace the .toml arguments"""
        default_args = vars(default_args)
        cmdl_args = vars(cmdl_args)
        changed_args = {}

        for key, value in default_args.items():
            if cmdl_args[key] != value:
                changed_args[key] = cmdl_args[key]

        return changed_args


    def parse_args(self, combined_section="general"):
        """Parse the arguments from the command line and the configuration file.
        If a section name is provided, only the arguments in that section will be parsed from the .toml file"""
        default_args, cmdl_args = self._extract_args()

        # These are the default arguments options updated by the command line
        if not cmdl_args.config:
            return cmdl_args

        # If a config file is passed, upodate the cmdl args with the config file unless
        # the argument is already specified in the command line
        try:
            config = toml.load(cmdl_args.config)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Config file {} not found".format(cmdl_args.config)
            )

        changed_args = self._find_changed_args(default_args, cmdl_args)
        if cmdl_args.section:
            try:
                section_name = cmdl_args.section        
                section_config = config[section_name]                
            except KeyError:
                raise KeyError(
                    "Section {} not found in config file".format(section_name)
                )

        else:
            section_config = config

        # Integrate args from a section that counts for all experiments
        if config.get(combined_section):
            section_config.update(config[combined_section])
            
        for key, value in section_config.items():
            if key not in default_args:
                # raise ValueError(
                    # "Warning: key '{}' in config file {} not found in argparser".format(
                        # key, default_args.config
                    # )
                # )
                continue
            # If the key has been passed in the command line, do not overwrite the 
            # command line argument with the toml argument, but vice versa.
            if key in changed_args:
                section_config[key] = changed_args[key]
            else:
                setattr(cmdl_args, key, value)

        return cmdl_args


# ------- Helper Functions for images ------- #


def img_to_uint8(img):
    """Convert an image to uint8"""
    img = (img - img.min()) * (1 / (img.max() - img.min()) * 255)
    if isinstance(img, torch.Tensor):
        img = img.type(torch.uint8)
    else:
        img = img.astype(np.uint8)

    return img


def load_image_data(img_root, filter_behavior=False, filter_plus=False):
    """Load image data from a folder"""
    dataset = ImageDataset(img_root=img_root, out_path="", transforms=get_image_transforms())
    assert len(dataset) > 0, "No images found in the image root"
    
    image_paths = dataset.images
    indices = np.arange(len(image_paths))

    class_names = [os.path.basename(img) for img in image_paths]

    if filter_behavior:
        indices = np.array(
        [i for i, img in enumerate(class_names) if "01b" in img]
        )    
    if filter_plus:
        indices = np.array(
        [i for i, img in enumerate(class_names) if "plus" in img]
        )
    
    image_paths = np.array(image_paths)[indices]

    return image_paths, indices


def get_image_transforms():
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=256),
            torchvision.transforms.CenterCrop(size=(224, 224)),
            torchvision.transforms.ToTensor(),
        ]
    )
    return transforms


# ------- Helper Functions for embeddings  ------- #


# Determine the cosine similarity between two vectors in pytorch
def cosine_similarity(embedding_i, embedding_j):
    if isinstance((embedding_i, embedding_j), torch.Tensor):
        return torch.nn.functional.cosine_similarity(embedding_i, embedding_j)
    else:
        return np.dot(embedding_i, embedding_j) / (
            np.linalg.norm(embedding_j) * np.linalg.norm(embedding_j)
        )


def load_deepnet_activations(activation_path, center=False, zscore=False, to_torch=False, relu=True):
    """Load activations from a .npy file"""
    # Check that not both center and zscore are true
    if center and zscore:
        raise ValueError("Cannot center and zscore activations at the same time")
    activation_path = glob.glob(os.path.join(activation_path, "*.npy"), recursive=True)
    if len(activation_path) > 1:
        raise ValueError("More than one .npy file found in the activation path")
    activation_path = activation_path[0]
    if activation_path.endswith("npy"):
        with open(activation_path, "rb") as f:
            act = np.load(f)
    else:
        act = np.loadtxt(activation_path)
    # We also add the positivity constraint here when loading activities!
    act = transform_activations(act, zscore=zscore, center=center, relu=relu)
    if to_torch:
        act = torch.from_numpy(act)

    return act

def transform_activations(act, zscore=False, center=False, relu=False):
    """Transform activations"""
    if center and zscore:
        raise ValueError("Cannot center and zscore activations at the same time")
    if relu:
        act = relu_embedding(act)
    # We standardize or center AFTER the relu. neg vals. are then meaningful
    if center:
        act = center_activations(act)
    if zscore:
        act = zscore_activations(act)

    return act

def center_activations(act):
    return act - act.mean(axis=0)

def zscore_activations(act, dim=0, eps=1e-8):
    std = np.std(act, axis=dim) + eps
    mean = np.mean(act, axis=dim)
    return (act - mean) / std

def relu_embedding(W):
    return np.maximum(0, W)


def create_path_from_params(path, *args):
    """Create a path if it does not exist"""
    import os
    base_path = os.path.dirname(os.path.dirname(path))
    out_path = os.path.join(base_path, *args)
    try:
        os.makedirs(out_path, exist_ok=True)
    except OSError as os:
        raise OSError("Error creating path {}: {}".format(path, os))

    return out_path


def remove_zeros(W, eps=0.1):
    w_max = np.max(W, axis=1)
    W = W[np.where(w_max > eps)]
    return W


def transform_params(weights, scale, relu=True):
    """We transform by (i) adding a positivity constraint and the sorting in descending order"""

    if relu:
        weights = relu_embedding(weights)
    sorted_dims = np.argsort(-np.linalg.norm(weights, axis=0, ord=1))

    weights = weights[:, sorted_dims]
    scale = scale[:, sorted_dims]
    d1, d2 = weights.shape
    # We transpose so that the matrix is always of shape (n_images, n_dims)
    if d1 < d2:
        weights = weights.T
        scale = scale.T

    return weights, scale, sorted_dims


def load_sparse_codes(path, weights=None, vars=None, with_dim=False, with_var=False, relu=True, zscore=False):
    """Load sparse codes from a directory. Can either be a txt file or a npy file or a loaded array of shape (n_images, n_dims)"""
    if weights is not None and vars is not None:
        assert isinstance(weights, np.ndarray) and isinstance(vars, np.ndarray), "Weights and var must be numpy arrays"
    elif isinstance(path, str):
        if path.endswith(".txt"):
            try:
                weights = np.loadtxt(path.replace("q_var", "q_mu"))
                vars = np.loadtxt(path.replace("q_mu", "q_var"))                  
            except OSError:
                raise OSError("Error loading sparse codes from path {}".format(path))

        elif path.endswith(".npz"):
            params = np.load(path)
            weights = params["pruned_q_mu"]
            vars = params["pruned_q_var"]

    elif isinstance(path, np.lib.npyio.NpzFile):
        weights = path["pruned_q_mu"]
        vars = path["pruned_q_var"]
    
    else:
        raise ValueError("Weights or Vars must be a .txt file path or as numpy array or .npz file")
    
    if "embedding" in os.path.basename(path):
        weights = np.loadtxt(path)
    
    weights, vars, sorted_dims = transform_params(weights, vars, relu=relu)

    if zscore:
        weights = zscore_activations(weights)
        vars = zscore_activations(vars)

    if with_dim:
        if with_var:
            return weights, vars, sorted_dims
        else:
            return weights, sorted_dims
    else:
        if with_var:
            return weights, vars
        else:
            
            return weights


# ------- Helper Functions for RSMs  ------- #


def fill_diag(rsm):
    """Fill main diagonal of the RSM with ones"""
    assert np.allclose(rsm, rsm.T), "\nRSM is required to be a symmetric matrix\n"
    rsm[np.eye(len(rsm)) == 1.0] = 1

    return rsm


def compute_rdm(X, method="correlation"):
    assert method in ["correlation", "euclidean"]
    if method == "euclidean":
        rdm = squareform(pdist(X, metric="euclidean"))
    else:
        rsm = correlation_matrix(X)
        rdm = 1 - rsm

    return rdm


def compute_rsm(X, method="correlation"):
    assert method in ["correlation", "euclidean"]
    if method == "euclidean":
        rdm = squareform(pdist(X, metric="euclidean"))
        rsm = 1 - rdm
    else:
        rsm = correlation_matrix(X)

    return rsm


def correlation_matrix(F, a_min=-1.0, a_max=1.0):
    """Compute dissimilarity matrix based on correlation distance (on the matrix-level)."""
    F_c = F - F.mean(axis=1)[:, np.newaxis]
    cov = F_c @ F_c.T
    # compute vector l2-norm across rows
    l2_norms = np.linalg.norm(F_c, axis=1)
    denom = np.outer(l2_norms, l2_norms)

    corr_mat = (cov / denom).clip(min=a_min, max=a_max)

    return corr_mat


def correlate_rsms(rsm_a, rsm_b, correlation="pearson"):
    """Correlate the lower triangular parts of two rsms"""
    rsm_a = fill_diag(rsm_a)
    rsm_b = fill_diag(rsm_b)
    triu_inds = np.triu_indices(len(rsm_a), k=1)
    corr_func = (
        scipy.stats.pearsonr if correlation == "pearson" else scipy.stats.spearmanr
    )
    rho = corr_func(rsm_a[triu_inds], rsm_b[triu_inds])[0]

    return rho


@njit(parallel=True, fastmath=True)
def matmul(A, B):
    i, k = A.shape
    k, j = B.shape
    C = np.zeros((i, j))
    for i in prange(i):
        for j in prange(j):
            for k in prange(k):
                C[i, j] += A[i, k] * B[k, j]
    return C



# ------- Helper Functions for Probability Densities  ------- #


def normal_pdf(X, loc, scale):
    """Probability density function of a normal distribution."""
    return (
        torch.exp(-((X - loc) ** 2) / (2 * scale.pow(2)))
        / scale
        * math.sqrt(2 * math.pi)
    )


def log_normal_pdf(X, loc, scale):
    LOG_NORMAL_CONST = torch.sqrt(torch.tensor(2 * math.pi))
    """Calculate the probability density function of the log-normal distribution"""
    X = F.relu(X) + 1e-12
    const = 1 / (X * scale * LOG_NORMAL_CONST)
    numerator = -((torch.log(X) - loc) ** 2)
    denominator = 2 * scale**2
    pdf = const * torch.exp(numerator / denominator)

    return pdf