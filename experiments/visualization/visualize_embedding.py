#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main visualization of embedding dimension based on sorted weights"""

import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np
import cv2
from object_dimensions import ExperimentParser
from object_dimensions.utils.utils import load_sparse_codes, load_image_data


parser = ExperimentParser(description="Visualize embedding")
parser.add_argument(
    "--embedding_path", type=str, default="./weights", help="path to weights directory"
)
parser.add_argument(
    "--img_root",
    type=str,
    default="./data/reference_images",
    help="Path to image root directory",
)
parser.add_argument(
    "--filter_behavior",
    default=False,
    action="store_true",
    help="If to filter the learned embedding by the behavior images",
)
parser.add_argument(
    "--filter_plus",
    default=False,
    action="store_true",
    help="If to filter the learned embedding by the behavior images",
)
parser.add_argument(
    "--per_dim", default=False, action="store_true", help="Plots per dimension if true"
)


parser.add_argument(
    "--behav-experiment", default=False, action="store_true", help="Plots large images for behavior experiment"
)

def plot_dim_3x3(images, codes, dim, top_k=10):
    # Check if codes is 2d or 1d
    if len(codes.shape) == 1:
        weight = codes
    else:
        weight = codes[:, dim]

    top_k = 9
    top_k_samples = np.argsort(-weight)[:top_k]  # this is over the image dimension
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)
    
    for k, sample in enumerate(top_k_samples):
        ax = axes[k // 3, k % 3]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        img = io.imread(images[sample])
        
        ax.imshow(img)
    return fig


def plot_large(images, codes, dim, top_k=16):
    # Check if codes is 2d or 1d
    if len(codes.shape) == 1:
        weight = codes
    else:
        weight = codes[:, dim]

    top_k_samples = np.argsort(-weight)[:top_k]  # this is over the image dimension
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)
    for k, sample in enumerate(top_k_samples):
        ax = axes[k // 4, k % 4]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        print(images[sample])
        img = io.imread(images[sample])
        ax.imshow(img)
    return fig


def plot_dim(images, codes, dim, top_k=10):
    # Check if codes is 2d or 1d
    if len(codes.shape) == 1:
        weight = codes
    else:
        weight = codes[:, dim]

    top_k_samples = np.argsort(-weight)[:top_k]  # this is over the image dimension
    fig, axes = plt.subplots(2, 5, figsize=(5, 2))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)
    for k, sample in enumerate(top_k_samples):
        ax = axes[k // 5, k % 5]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        img = io.imread(images[sample])
        ax.imshow(img)
    return fig


def plot_per_dim(args):
    base_path = os.path.dirname(os.path.dirname(args.embedding_path))
    results_path = os.path.join(base_path, "analyses", "per_dim")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    W = load_sparse_codes(args.embedding_path)
    images, indices = load_image_data(
        args.img_root,
        filter_behavior=args.filter_behavior,
        filter_plus=args.filter_plus,
    )

    if W.shape[0] != len(images):
        W = W[indices]
    append = (
        "_plus" if args.filter_plus else "_behavior" if args.filter_behavior else ""
    )

    print("Shape of weight Matrix", W.shape)

    top_k = 10
    n_dims = W.shape[1]

    for dim in range(n_dims):

        if args.behav_experiment:
            behav_path = os.path.join(base_path, "analyses", "behavior_experiment", "images")
            if not os.path.exists(behav_path):
                os.makedirs(behav_path)
            fig_large = plot_large(images, W, dim, top_k=16)
            for ext in ["png", "pdf"]:
                fname = os.path.join(behav_path, f"{dim}_topk{append}_large.{ext}")
                fig_large.savefig(fname, dpi=150, bbox_inches="tight", pad_inches=0)
            plt.close(fig_large)
            print(f"Done plotting for dim {dim} large")
            continue


        fig_5x2 = plot_dim(images, W, dim, top_k)
        fig_3x3 = plot_dim_3x3(images, W, dim, top_k)

        # fig.suptitle("Dimension: {}".format(dim))
        out_path = os.path.join(results_path, f"{dim:02d}")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    
        for ext in ["png", "pdf"]:
            fname = os.path.join(out_path, f"{dim}_topk{append}_5x2.{ext}")
            fig_5x2.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0)
            fname = os.path.join(out_path, f"{dim}_topk{append}_3x3.{ext}")
            fig_3x3.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig_5x2)
        plt.close(fig_3x3)
        print(f"Done plotting for dim {dim}")

        


    


def plot_dimensions(args):
    W = load_sparse_codes(args.embedding_path)
    images, indices = load_image_data(
        args.img_root,
        filter_behavior=args.filter_behavior,
        filter_plus=args.filter_plus,
    )

    images, indices = load_image_data(
        args.img_root,
        filter_behavior=args.filter_behavior,
        filter_plus=args.filter_plus,
    )

    if W.shape[0] != len(images):
        W = W[indices]

    print("Shape of weight Matrix", W.shape)
    W = W.T

    top_k = 12
    n_rows = W.shape[0]
    n_cols = top_k
    fig = plt.figure(figsize=(int(top_k * 1.5), int(n_rows * 1.6)))
    gs1 = gridspec.GridSpec(n_rows, n_cols)
    gs1.update(wspace=0.025, hspace=0)  # set the spacing between axes.

    for j, w_j in enumerate(W):
        top_k_samples = np.argsort(-w_j)[:top_k]  # this is over the image dimension
        for k, sample in enumerate(top_k_samples):
            # take the axes j and k from the grid
            ax = plt.subplot(gs1[j * n_cols + k])
            ax.imshow(io.imread(images[sample]))
            ax.set_xticks([])
            ax.set_yticks([])

            # Set a ylabel per row
            if k == 0:
                ax.set_ylabel(f"{j}", rotation=0, va="center", ha="right", fontsize=14)

    # Save figure in predefined embedding path directory
    base_path = os.path.dirname(os.path.dirname(args.embedding_path))
    filename = os.path.basename(args.embedding_path)
    epoch = filename.split("_")[-1].split(".")[0]
    out_path = os.path.join(base_path, "analyses")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    fname = os.path.join(out_path, "all_dimensions{}_epoch_{}.png")

    if args.filter_behavior:
        fname = fname.format("_filtered_behavior", epoch)
    if args.filter_plus:
        fname = fname.format("_filtered_plus", epoch)
    else:
        fname = fname.format("", epoch)
    fig.savefig(fname, dpi=50, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.per_dim:
        plot_per_dim(args)
    else:
        plot_dimensions(args)
