#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
from PIL import Image


class ImageDataset:
    """ Loads all images in a given directory and stores its filenames and paths """
    def __init__(self, img_root, out_path, transforms=None):
        self.img_root = img_root
        self.out_path = out_path
        self._find_image_paths()
        self.transforms = transforms

    def _find_image_paths(self):
        """ Find all images ending with .jpg in image_root recursively """
        path = os.path.join(self.img_root, "**", "*.jpg")
        self.samples = glob.glob(path, recursive=True)
        self.samples = sorted(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform_image(img)
        return img

    def transform_image(self, img):
        img = self.transforms(img)
        return img

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def images(self):
        return self.samples