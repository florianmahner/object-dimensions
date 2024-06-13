#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import torch
import torchvision

from PIL import Image
from pathlib import Path
from typing import Optional, List

import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt


# ------------------ image related data processing ------------------ #


class ImageDataset(object):
    """Loads all images in a given directory and stores its filenames and paths"""

    def __init__(
        self,
        img_root: str,
        out_path: str,
        transforms: Optional[T.Compose] = None,
    ) -> None:
        self.img_root = img_root
        self.out_path = out_path
        self._find_image_paths()
        self.transforms = transforms

    def _find_image_paths(self) -> None:
        """Find all images ending with .jpg in image_root recursively"""
        path = os.path.join(self.img_root, "**", "*.jpg")
        self.samples = glob.glob(path, recursive=True)
        # Sort samples by parent directory and then by filename within each directory
        self.samples = sorted(
            self.samples, key=lambda x: (Path(x).parent, Path(x).name)
        )

    def __getitem__(self, idx: int) -> Image.Image:
        img_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform_image(img)
        return img

    def transform_image(self, img: Image.Image) -> Image.Image:
        img = self.transforms(img)
        return img

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def images(self) -> List:
        return self.samples


def img_to_uint8(img):
    """Convert an image to uint8"""
    img = (img - img.min()) * (1 / (img.max() - img.min()) * 255)
    if isinstance(img, torch.Tensor):
        img = img.type(torch.uint8)
    else:
        img = img.astype(np.uint8)

    return img


def load_data(human_path, feature_path, img_root):
    features = load_deepnet_activations(feature_path, relu=True, center=True)
    human_embedding = load_sparse_codes(human_path, relu=True)
    indices = load_image_data(img_root, filter_behavior=True)[1]
    features = features[indices]
    return features, human_embedding


def load_image_data(
    img_root,
    filter_behavior=False,
    filter_plus=False,
):
    """Load image data from a folder"""
    dataset = ImageDataset(
        img_root=img_root, out_path="", transforms=get_image_transforms()
    )
    assert len(dataset) > 0, "No images found in the image root"

    image_paths = dataset.images
    indices = np.arange(len(image_paths))

    class_names = [os.path.basename(img) for img in image_paths]

    if filter_behavior:
        indices = np.array([i for i, img in enumerate(class_names) if "01b" in img])
    if filter_plus:
        indices = np.array([i for i, img in enumerate(class_names) if "plus" in img])

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


def save_figure(fig, out_path, dpi=300, extensions=["pdf", "png"], **kwargs):
    """Save a figure to disk"""
    for ext in extensions:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format=ext, **kwargs)
    plt.close(fig)


# ------------------ embedding and feature processing ------------------ #


def load_deepnet_activations(
    activation_path, center=False, zscore=False, to_torch=False, relu=True
):
    """Load activations from a .npy file"""
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


def create_results_path(embedding_path, *args, base_path="./results"):
    """Create a path if it does not exist. Each argument is a subdirectory in the path."""
    import os

    try:
        model_name = Path(embedding_path).parts[-2]
    except IndexError:
        raise ValueError("Invalid embedding path: unable to extract model name")
    out_path = os.path.join(base_path, "experiments", model_name, *args)

    try:
        os.makedirs(out_path, exist_ok=True)
    except OSError as os:
        raise OSError("Error creating path {}: {}".format(out_path, os))

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


def load_sparse_codes(
    path, weights=None, vars=None, with_dim=False, with_var=False, relu=True
):
    """Load sparse codes from a directory. Can either be a txt file or a npy file or a loaded array of shape (n_images, n_dims)"""
    if weights is not None and vars is not None:
        assert isinstance(weights, np.ndarray) and isinstance(
            vars, np.ndarray
        ), "Weights and var must be numpy arrays"

    file = glob.glob(os.path.join(path, "parameters.npz"))
    if len(file) > 0:
        params = np.load(os.path.join(path, "parameters.npz"))
        if params["method"] == "variational":
            weights = params["pruned_q_mu"]
            vars = params["pruned_q_var"]
        else:
            weights = params["pruned_weights"]
            vars = np.zeros_like(weights)
    elif isinstance(path, str):
        if path.endswith(".txt"):
            # Check if q_mu or q_var in path
            if "q_mu" or "q_var" in path:
                try:
                    weights = np.loadtxt(path.replace("q_var", "q_mu"))
                    vars = np.loadtxt(path.replace("q_mu", "q_var"))
                except OSError:
                    raise OSError(
                        "Error loading sparse codes from path {}".format(path)
                    )
            else:
                if "embedding" in os.path.basename(path):
                    weights = np.loadtxt(path)
                    vars = None

        elif path.endswith(".npz"):
            params = np.load(path)
            if params["method"] == "variational":
                weights = params["pruned_q_mu"]
                vars = params["pruned_q_var"]

            else:
                weights = params["pruned_weights"]
                vars = np.zeros_like(weights)

    elif isinstance(path, np.lib.npyio.NpzFile):
        if path["method"] == "variational":
            weights = path["pruned_q_mu"]
            vars = path["pruned_q_var"]
        else:
            weights = path["pruned_weights"]
            vars = np.zeros_like(weights)

    else:
        raise ValueError(
            "Weights or Vars must be a .txt file path or as numpy array or .npz file"
        )

    if weights is None:
        raise FileNotFoundError("No embedding found in the path {}".format(path))

    weights, vars, sorted_dims = transform_params(weights, vars, relu=relu)
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
