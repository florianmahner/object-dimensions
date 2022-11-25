#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import torch
import os
import math
import glob
import scipy.stats
import toml
from dataclasses import dataclass

import sys

import torch.nn.functional as F

import numpy as np

from numba import njit, prange
from scipy.spatial.distance import pdist, squareform
from deep_embeddings.utils.image_dataset import ImageDataset   
import torchvision

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=256),
    torchvision.transforms.CenterCrop(size=(224, 224)),
    torchvision.transforms.ToTensor()])

LOG_NORMAL_CONST = torch.sqrt(torch.tensor(2 * math.pi))


class ExperimentParser:
    def __init__(self, *args, **kwargs):
        self.parser = argparse.ArgumentParser(*args, **kwargs)
        self.parser.add_argument('--config', type=str, default="", help="""Path to the configuration file. 
                        This configuration file can store and replace all default argparse arguments""")

    def _update_toml_with_command_line(self, toml_config, cmd_line_args):
        """ Update the configuration file with the command line arguments """
        for i, command in enumerate(cmd_line_args):
            command = command.replace("-", "")
            if command in toml_config:
                # If this is the case then a boolean values has been passed and we store true
                if i == len(cmd_line_args)-1:
                    toml_config[command] = True
                elif cmd_line_args[i + 1] in toml_config[command]:
                    toml_config[command] = True
                else:
                    toml_config[command] = cmd_line_args[i + 1]

        return toml_config

    def _parse_from_config(self, args):
        """ Parse arguments from a configuration file """
        if not args.config:
            return args
        config = toml.load(args.config)
        command_line_args = sys.argv[1:]
        config = self._update_toml_with_command_line(config, command_line_args)
        for key, value in config.items():
            if key not in args:
                raise ValueError("Warning: key '{}' in config file {} not found in argparser".format(key, args.config))
            else:
                setattr(args, key, value)
        return args

    def add_argument(self, *args, **kwargs):
        return self.parser.add_argument(*args, **kwargs)

    def parse_args(self):
        args = self.parser.parse_args()    
        args = self._parse_from_config(args)        
        return args


def img_to_uint8(img):
    img = ((img - img.min()) * (1/(img.max() - img.min()) * 255))
    if isinstance(img, torch.Tensor):
        img = img.type(torch.uint8)
    else:
        img = img.astype(np.uint8)

    return img

def load_image_data(img_root):
    """ Load image data from a folder """
    dataset = ImageDataset(img_root=img_root, out_path='', transforms=transforms)
    assert len(dataset) > 0, "No images found in the image root"
    idx2obj = None
    obj2idx = None
    images = dataset.images

    return idx2obj, obj2idx, images

def filter_embedding_by_behavior(embedding, image_paths):
    print("Select only behavior images to visualize the embedding")
    class_names = [os.path.basename(img) for img in image_paths]
    behavior_indices = np.array([i for i, img in enumerate(class_names) if '01b' in img])
    image_paths = np.array(image_paths)[behavior_indices]
    embedding = embedding[behavior_indices]

    return embedding, image_paths

def filter_embedding_by_plus(embedding, image_paths):
    print("Select only plus images to visualize the embedding")
    class_names = [os.path.basename(img) for img in image_paths]
    behavior_indices = np.array([i for i, img in enumerate(class_names) if 'plus' in img])
    image_paths = np.array(image_paths)[behavior_indices]
    embedding = embedding[behavior_indices]

    return embedding, image_paths

# Determine the cosine similarity between two vectors in pytorch
def cosine_similarity(embedding_i, embedding_j):
    if isinstance((embedding_i, embedding_j), torch.Tensor):
        return torch.nn.functional.cosine_similarity(embedding_i, embedding_j)
    else:
        return np.dot(embedding_i, embedding_j) / (np.linalg.norm(embedding_j) * np.linalg.norm(embedding_j))
    
def relu_embedding(W):
    return np.maximum(0, W)

def remove_zeros(W, eps=.1):
    w_max = np.max(W, axis=1)
    W = W[np.where(w_max > eps)]
    return W

def transform_weights(weights, relu=True):
    """ We transform by (i) adding a positivity constraint and the sorting in descending order"""
    if relu:
        weights = relu_embedding(weights) 
    weights = remove_zeros(weights)
    
    sorted_dims = np.argsort(-np.linalg.norm(weights, axis=0, ord=1))
    weights = weights[:, sorted_dims]

    d1, d2 = weights.shape
    # We transpose so that the matrix is always of shape (n_images, n_dims)
    if d1 < d2:
        weights = weights.T

    return weights, sorted_dims


def load_deepnet_activations(activation_path, center=False, to_torch=False):
    """ Load activations from a .npy file """
    activation_path = glob.glob(os.path.join(activation_path, "*.npy"), recursive=True)
    if len(activation_path) > 1:
        raise ValueError("More than one .npy file found in the activation path")
    activation_path = activation_path[0]
    if activation_path.endswith('npy'):
        with open(activation_path, 'rb') as f:
            act = np.load(f)
    else:
        act = np.loadtxt(activation_path)
    if center:
        center_activations(act)
    # We also add the positivity constraint here when loading activities!
    act = np.maximum(0, act)
    if to_torch:
        act = torch.from_numpy(act)
    
    return act


def center_activations(act):
    return act - act.mean(axis=0)

def load_sparse_codes(path, with_dim=False, relu=True): 
    weights = np.loadtxt(os.path.join(path))
    weights, sorted_dim = transform_weights(weights, relu=relu)
    if with_dim:
        return weights, sorted_dim
    else:
        return weights
    

def fill_diag(rsm):
    """ Fill main diagonal of the RSM with ones """
    assert np.allclose(rsm, rsm.T), '\nRSM is required to be a symmetric matrix\n'
    rsm[np.eye(len(rsm)) == 1.] = 1

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

def correlation_matrix(F, a_min= -1., a_max= 1.):
    ''' Compute dissimilarity matrix based on correlation distance (on the matrix-level). '''
    F_c = F - F.mean(axis=1)[:, np.newaxis]
    cov = F_c @ F_c.T
    # compute vector l2-norm across rows
    l2_norms = np.linalg.norm(F_c, axis=1)
    denom = np.outer(l2_norms, l2_norms)
    corr_mat = (cov / denom).clip(min=a_min, max=a_max)
    
    return corr_mat

def correlate_rsms(rsm_a, rsm_b, correlation='pearson'):
    """ Correlate the lower triangular parts of two rsms"""
    rsm_a = fill_diag(rsm_a)
    rsm_b = fill_diag(rsm_b)
    triu_inds = np.triu_indices(len(rsm_a), k=1)
    corr_func = scipy.stats.pearsonr if correlation == 'pearson' else scipy.stats.spearmanr
    rho = corr_func(rsm_a[triu_inds], rsm_b[triu_inds])[0]

    return rho

# NOTE This seems to be wrong -> not the same results as obtained from torch.Distributions or scipy.stats
# def normal_pdf(X, loc, scale):
    # gauss_pdf = torch.exp(-((X - loc) ** 2) / (2 * scale.pow(2))) / scale * math.sqrt(2 * math.pi)

    # return gauss_pdf

def normal_pdf(X, loc, scale):
    """Probability density function of a normal distribution."""
    return (
        torch.exp(-((X - loc) ** 2) / (2 * scale.pow(2)))
        / scale
        * math.sqrt(2 * math.pi)
    )

# def log_normal_pdf(X, loc, scale):
#     return torch.distributions.Normal(loc, scale).log_prob(X)

def log_normal_log_pdf(X, loc, scale):
    
    """ Calculate the probability density function of the log-normal distribution """
    X = F.relu(X) + 1e-12
    log_pdf = torch.distributions.LogNormal(loc, scale).log_prob(X)

    # const = 1 / (X * scale * LOG_NORMAL_CONST)
    # const = torch.log(const)

    # term1 = -((torch.log(X) - loc)**2) / (2 * scale**2)
    # log_pdf = const + term1

    # breakpoint()

    return log_pdf


def log_normal_pdf(X, loc, scale):
    """ Calculate the probability density function of the log-normal distribution """
    X = F.relu(X) + 1e-12
    const = 1 / (X * scale * LOG_NORMAL_CONST)
    numerator = -((torch.log(X) - loc)**2) 
    denominator = 2 * scale**2
    pdf = const * torch.exp(numerator / denominator)

    return pdf



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
