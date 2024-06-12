#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import torch
import torchvision

from PIL import Image
from pathlib import Path
from typing import Optional, List
from objdim.utils import load_deepnet_activations, load_sparse_codes

import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np


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
