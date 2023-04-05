#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import torch
import os
import pickle

import numpy as np
from thingsvision import Extractor
from thingsvision.utils.data import ImageDataset

from object_dimensions.utils.searchlight_utils import mask_img
from object_dimensions.analyses.image_generation.latent_predictor import LatentPredictor


class Config:
    latent_path = "./"
    regression_path = "../sparse_codes/sparse_code_predictions"
    feature_path = "/LOCAL/fmahner/THINGS/vgg_bn_features_12/features.npy"

    model_name = "vgg16_bn"
    module_name = "classifier.3"
    latent_version = "sampled"

    image_root = "/LOCAL/fmahner/THINGS/image_data/images12"

    analysis = "regression"  # or classification / feature extraction
    latent_dim = 7  # do for specific dim

    window_size = 20
    stride = 1
    top_k = 2
    truncation = 0.4

    device = "cuda:0"
    seed = 42


def searchlight_(img, regression_predictor, window_size, stride=1, latent_dim=1):
    H, W = img.shape[-2:]
    diffs = torch.zeros((H, W))

    # NOTE we could also index previous feature mat but dont need to do this for now!
    # TODO Check if I really only want to do this for one dimension at a time?
    dim_original = regression_predictor.predict_codes_from_img(img)
    dim_original = dim_original[latent_dim]

    with torch.no_grad():
        for i in np.arange(0, H, stride):
            print(
                f"\n...Currently masking image centered around row position {i}.",
                end="\r",
            )
            for j in np.arange(0, W, stride):
                img_copy = img.clone()

                # mask a window of size ws x ws centered around position (i, j)
                masked_img = mask_img(img_copy, i, j, window_size)
                dim_masked = regression_predictor.predict_codes_from_img(masked_img)
                dim_masked = dim_masked[latent_dim]

                p_explained = torch.abs(dim_original - dim_masked) / (
                    dim_original + 1e-12
                )  # NOTE add this to avoid division by zero
                diffs[i : i + stride, j : j + stride] = p_explained

    diffs = diffs.cpu().numpy()
    return diffs


def find_topk_images(predictor, features, top_k, latent_dim):
    """Find the top k images that maximally activate each dimension based on vgg features. Returns indices"""

    dimensions = predictor.predict_codes_from_features(features)

    top_k_images = torch.argsort(-dimensions, dim=0)[:top_k]
    top_k_images = top_k_images[
        :, latent_dim
    ]  # take the topk samples for a specific dimension!

    return top_k_images


def search_image_spaces(
    regression_predictor,
    dataset,
    features,
    top_k=16,
    window_size=20,
    stride=1,
    latent_dim=1,
    device="cpu",
):

    top_k_images = find_topk_images(regression_predictor, features, top_k, latent_dim)

    out_path = os.path.join("searchlights", "dnn", "top_k", f"{latent_dim:02d}")
    if not os.path.exists(out_path):
        print("\n...Creating directories.\n")
        os.makedirs(out_path)

    for i, k in enumerate(top_k_images):
        img = dataset[k].to(device).unsqueeze(0)
        diffs = searchlight_(img, regression_predictor, window_size, stride, latent_dim)
        img = img.T.cpu().numpy().squeeze()
        save_dict = {"diffs": diffs, "img": img}

        # store save dict as pickle file in directory
        with open(
            os.path.join(
                out_path, f"./searchlight_ws_{window_size}_stride_{stride}_{i}.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(save_dict, f)

        # np.save(os.path.join(out_path, f'./searchlight_ws_{window_size}_stride_{stride}_{i}.npy'), save_dict)


if __name__ == "__main__":

    # NOTE wine glass index  = (21919) in images12 imagedataset!

    cfg = Config

    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)

    extractor = Extractor(
        model_name=cfg.model_name, pretrained=True, device=device, source="torchvision"
    )
    dataset = ImageDataset(
        root=cfg.image_root,
        out_path="",
        backend=extractor.backend,
        transforms=extractor.get_transformations(),
    )

    predictor = LatentPredictor(
        cfg.model_name, cfg.module_name, device, cfg.regression_path
    )

    features = np.load(cfg.feature_path)

    predictor.to(device)
    features = torch.from_numpy(features).to(device)

    search_image_spaces(
        predictor,
        dataset,
        features,
        cfg.top_k,
        cfg.window_size,
        cfg.stride,
        cfg.latent_dim,
        device,
    )
