#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import h5py
import pickle

from pathlib import Path
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from skimage.transform import resize
from tomlparse import argparse

from object_dimensions.utils import (
    load_image_data,
    load_sparse_codes,
)
from object_dimensions.latent_predictor import LatentPredictor
import random
import torch
import os
import pickle
import numpy as np
from thingsvision import get_extractor
from collections import defaultdict
from PIL import Image
from tomlparse import argparse



QUERIES = [
    ("wine_01b", [2, 56, 22, 35, 15, 51]),
    ("flashlight_01b", [25, 24, 44, 35, 51]),
    ("basketball_plus", [37, 28, 39]),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Searchlight analysis for one image.")
    parser.add_argument(
        "--embedding_path", type=str, help="Path to the embedding file."
    )
    parser.add_argument(
        "--img_root",
        type=str,
        default="./data/images",
        help="Path to the all images used for the embedding.",
    )
    parser.add_argument(
        "--analysis",
        type=str,
        default="regression",
        help="Type of analysis to perform.",
    )
    parser.add_argument(
        "--model_name", type=str, default="vgg16_bn", help="Name of the model to use."
    )
    parser.add_argument(
        "--module_name",
        type=str,
        default="classifier.3",
        help="Name of the module to use.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed to use for the searchlight."
    )
    return parser.parse_args()


class RISE(nn.Module):
    """Adapted from https://github.com/eclique/RISE"""

    def __init__(self, model, input_size, n_cls, gpu_batch=100):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch
        self.n_cls = n_cls

    def generate_masks(self, N, s, p1, savepath="masks.hdf5"):
        """
        p1: probability of setting a pixel to 1
        n: number of masks
        s: size of the mask
        """

        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype("float32")

        self.masks = np.empty((N, *self.input_size))

        for i in tqdm(range(N), desc="Generating filters"):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(
                grid[i], up_size, order=1, mode="reflect", anti_aliasing=False
            )[x : x + self.input_size[0], y : y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)

        save_dict = {
            "masks": self.masks,
            "N": N,
            "p1": p1,
        }

        # Save the dictionary to an HDF5 file
        with h5py.File(savepath, "w") as f:
            for key, value in save_dict.items():
                f.create_dataset(key, data=value)

        self.masks = torch.from_numpy(self.masks).float()
        self.mask_loader = MaskLoader(self.masks, self.gpu_batch)
        self.N = N
        self.p1 = p1

    def load_masks(self, filepath):
        with h5py.File(filepath, "r") as f:
            self.masks = f["masks"][:]
            self.p1 = f["p1"][()]
            self.N = f["N"][()]

        self.masks = torch.from_numpy(self.masks).float()
        self.mask_loader = MaskLoader(self.masks, self.gpu_batch)

    def forward(self, x):
        """Adapted this from the original code to work with a batch of masks"""

        tsfm = self.model.transforms
        x = tsfm(x).to(self.model.device)
        # x.requires_grad = True

        height, width = x.size()[-2:]
        sal = torch.zeros((self.n_cls, height, width), device=x.device)

        # with torch.no_grad():
        for i, masks_batch in enumerate(self.mask_loader):
            print(
                "Processing batch {}/{}".format(i + 1, len(self.mask_loader)),
                end="\r",
            )
            masks_batch = masks_batch.to(x.device)

            # Apply the batch of masks to the image
            stack = torch.mul(masks_batch, x.data)
            stack.requires_grad = True

            p_batch = self.model(stack, transform=False)[1]
            masks_batch_flat = masks_batch.view(masks_batch.size(0), height * width)

            sal_batch = torch.matmul(p_batch.data.transpose(0, 1), masks_batch_flat)
            sal_batch = sal_batch.view((self.n_cls, height, width))
            sal += sal_batch

        sal = sal / self.N / self.p1
        return sal


class MaskLoader:
    def __init__(self, masks, batch_size):
        self.masks = masks.float()
        self.N = len(self.masks)
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, self.N, self.batch_size):
            yield self.masks[i : i + self.batch_size]

    def __len__(self):
        return (self.N + self.batch_size - 1) // self.batch_size


def run():
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"

    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor = get_extractor(
        model_name=args.model_name, pretrained=True, device=device, source="torchvision"
    )
    images, indices = load_image_data(args.img_root)

    base_path = os.path.dirname(os.path.dirname(args.embedding_path))
    regression_path = os.path.join(base_path, "analyses", "sparse_codes")

    predictor = LatentPredictor(
        args.model_name, args.module_name, device, regression_path
    )
    predictor.to(device)

    images_plus, indices_plus = load_image_data(args.img_root, filter_plus=True)
    images, indices = load_image_data(args.img_root, filter_plus=False)
    sparse_codes = load_sparse_codes(args.embedding_path)
    sparse_codes = sparse_codes[indices]

    input_size = (224, 224)
    gpu_batch = 40
    n_cls = predictor.embedding_dim

    explainer = RISE(predictor, input_size, n_cls, gpu_batch)

    s = 8
    p1 = 0.1
    n_masks = 40_000

    maskspath = f"./data/masks/mask_{n_masks}_{s}_{p1}.hdf5"
    os.makedirs(os.path.dirname(maskspath), exist_ok=True)

    if not os.path.isfile(maskspath):
        explainer.generate_masks(N=n_masks, s=s, p1=p1, savepath=maskspath)
    else:
        explainer.load_masks(maskspath)
        print("Masks are loaded.")

    for img_idx, img in enumerate(images_plus):
        save_dict = defaultdict(list)
        name = Path(img).stem
        img = Image.open(img)

        tsfm = predictor.transforms
        img_vis = tsfm(img)

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_vis = img_vis.squeeze().numpy().transpose(1, 2, 0)
        img_vis = img_vis * std + mean

        img_vis = np.clip(img_vis, 0, 1)

        saliency = explainer(img).cpu().numpy()
        saliency = saliency.squeeze()
        latent_dims = np.argsort(-sparse_codes[img_idx])[:32]
        saliency = saliency[latent_dims]

        save_dict["img"] = img_vis
        save_dict["saliencies"] = saliency
        save_dict["dim"] = latent_dims  

        save_path = os.path.join(base_path, "analyses", "rise", f"{name}_rise.pkl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pickle.dump(save_dict, open(save_path, "wb"))


if __name__ == "__main__":
    run()
