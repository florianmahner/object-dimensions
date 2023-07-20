#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import torch
import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from thingsvision import get_extractor
from collections import defaultdict, Counter
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from tomlparse import argparse
from experiments.visualization.visualize_embedding import plot_dim_3x3
from object_dimensions.utils.utils import (
    img_to_uint8,
    load_image_data,
    load_sparse_codes,
)
from object_dimensions.utils.latent_predictor import LatentPredictor


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


def find_gradient_heatmap_(img, regression_predictor, latent_dim=1):
    """Extracts the gradients set by a hook in the regression predicto"""
    dim_original = regression_predictor(img, transform=True)[1]
    dim_original = dim_original[latent_dim]
    # Attach grad to leaf node
    img.requires_grad = True

    # Do a forward pass whilte preserving the graph
    with torch.enable_grad():
        dim_predict = regression_predictor(img, transform=True)[1]
        latent = dim_predict[latent_dim]
        latent.backward()
        gradients = regression_predictor.get_activations_gradient()

    # Pool  the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Get the activations of the last convolutional layer
    activations = regression_predictor.get_activations(img)[0].detach()

    # Idea: the sensitivity of activations to a target class can be
    # understood as the importance of the activation map to the class
    # (given by the gradient), hence we weight the activation maps
    # with the gradients
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.cpu().numpy()
    return heatmap


def search_image_spaces(
    base_path,
    regression_predictor,
    dataset,
    indices_plus,
    img_idx,
    image_name,
    sparse_codes,
    latent_dims=[1, 2, 3],
):
    out_path = os.path.join(base_path, "analyses", "grad_cam", "{}".format(image_name))
    if not os.path.exists(out_path):
        print("\n...Creating directories.\n")
        os.makedirs(out_path)

    img = Image.open(dataset[img_idx])

    reshape = T.Compose([T.Resize(800), T.CenterCrop(700)])
    img_vis = reshape(img)
    img_vis = np.array(img_vis)

    fig, ax = plt.subplots(frameon=False, dpi=300)
    ax.imshow(img_vis)
    ax.axis("off")
    path = os.path.join(out_path, "img.pdf")
    fig.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0,
    )

    save_dict = {"img": img_vis, "dim": latent_dims, "heatmaps": []}

    for dim in latent_dims:
        print(
            f"\n...Currently performing grad-cam analysis for latent dimension {dim}."
        )

        heatmap_grads = find_gradient_heatmap_(img, regression_predictor, dim)
        save_dict["heatmaps"].append(heatmap_grads)
        heatmap = heatmap_grads
        heatmap = img_to_uint8(heatmap)
        heatmap = cv2.resize(heatmap, (img_vis.shape[1], img_vis.shape[0]))

        # Invert heatmap so that 255 is 0 and 0 is 255
        heatmap = cv2.bitwise_not(heatmap)
        cmap = cv2.COLORMAP_JET
        heatmap_img = cv2.applyColorMap(heatmap, cmap)
        super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img_vis, 0.5, 0)

        fig, ax = plt.subplots(frameon=False, dpi=300)
        ax.imshow(heatmap, cmap="jet")
        ax.axis("off")
        path = os.path.join(out_path, "./{}_heatmap{}")
        fig.savefig(path.format(dim, ".pdf"), bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        fig, ax = plt.subplots(frameon=False, dpi=300)
        ax.imshow(super_imposed_img)
        ax.axis("off")
        path = os.path.join(out_path, "./{}_cam_superimposed{}")
        for ext in [".png", ".pdf"]:
            plt.savefig(path.format(dim, ext), bbox_inches="tight", pad_inches=0)

        plt.close(fig)

        images_plus = images[indices_plus]
        sparse_codes_plus = sparse_codes[indices_plus]
        fig_img = plot_dim_3x3(images_plus, sparse_codes_plus, dim, top_k=10)
        for ext in ["pdf"]:
            fname = os.path.join(out_path, f"{dim}_topk_plus.{ext}")
            fig_img.savefig(fname, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close(fig_img)

    # store save dict as pickle file in directory
    with open(os.path.join(out_path, f"./cam_all.pkl"), "wb") as f:
        pickle.dump(save_dict, f)


def find_topk_per_image(embedding, topk=8):
    """Finds the topk dimensions that activate each image in the embedding space.
    Filters these topk images to only include the most important images that are
    in many of the topk dimensions."""
    n_dim = embedding.shape[1]
    image_topk_counter = Counter()
    image_dims = defaultdict(list)
    topk_images = defaultdict(list)

    for dim in range(n_dim):
        codes = embedding[:, dim]
        topk_indices = list(np.argsort(-codes)[:topk])
        image_topk_counter.update(topk_indices)

        # Append each index to the list of topk dims for that image
        for idx in topk_indices:
            image_dims[idx].append(dim)

        # Create a list of topk images for that dimension
        topk_images[dim].append(topk_indices)

    most_common_topk = image_topk_counter.most_common(20)
    most_common_topk = [x[0] for x in most_common_topk]

    # Get the dict that has key = image_index, value = topk dims for that image
    topk_images_shared = defaultdict(list)
    for idx in most_common_topk:
        topk_images_shared[idx] = image_dims[idx]

    return topk_images_shared, topk_images


if __name__ == "__main__":
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

    for query, latent_dims in QUERIES:
        img_idx = [i for i, img in enumerate(images) if query in img][0]
        search_image_spaces(
            base_path,
            predictor,
            images,
            indices_plus,
            img_idx,
            query,
            sparse_codes,
            latent_dims,
        )
