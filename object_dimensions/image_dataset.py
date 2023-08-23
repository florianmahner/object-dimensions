#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
from PIL import Image
import torchvision.transforms as T

from typing import List, Optional


class ImageDataset:
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
        self.samples = sorted(self.samples)

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
