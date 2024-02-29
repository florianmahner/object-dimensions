#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main visualization of embedding dimension based on sorted weights"""

import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np

from tomlparse import argparse
from object_dimensions.utils import load_sparse_codes, load_image_data


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize embedding")
    parser.add_argument(
        "--embedding_path",
        type=str,
        default="./weights",
        help="path to weights directory",
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
        "--per_dim",
        default=False,
        action="store_true",
        help="Plots per dimension if true",
    )

    parser.add_argument(
        "--behav-experiment",
        default=False,
        action="store_true",
        help="Plots large images for behavior experiment",
    )
    return parser.parse_args()


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


def plot_dim_3x2(images, codes, dim, top_k=10):
    # Check if codes is 2d or 1d
    if len(codes.shape) == 1:
        weight = codes
    else:
        weight = codes[:, dim]

    top_k = 6
    top_k_samples = np.argsort(-weight)[:top_k]  # this is over the image dimension
    fig, axes = plt.subplots(2, 3, figsize=(6, 4))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)

    for k, sample in enumerate(top_k_samples):
        ax = axes[k // 3, k % 3]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        img = io.imread(images[sample])

        ax.imshow(img)
    return fig


def plot_dim_1x8(images, codes, dim, top_k=10):
    # Check if codes is 2d or 1d
    if len(codes.shape) == 1:
        weight = codes
    else:
        weight = codes[:, dim]

    top_k = 8
    top_k_samples = np.argsort(-weight)[:top_k]  # this is over the image dimension
    fig, axes = plt.subplots(1, 8, figsize=(8, 1))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.0)

    for k, sample in enumerate(top_k_samples):
        ax = axes[k % 8]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        img = io.imread(images[sample])

        ax.imshow(img)
    return fig


def plot_top_50(images, codes, dim, topk=50):
    # Check if codes is 2d or 1d
    if len(codes.shape) == 1:
        weight = codes
    else:
        weight = codes[:, dim]

    top_k = 50
    top_k_samples = np.argsort(-weight)[:top_k]  # this is over the image dimension
    fig, axes = plt.subplots(5, 10, figsize=(10, 5))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.0)

    for k, sample in enumerate(top_k_samples):
        ax = axes[k // 10, k % 10]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        img = io.imread(images[sample])

        ax.imshow(img)
    return fig


def plot_behavior(images, codes, dim, top_k=6):
    # Check if codes is 2d or 1d
    if len(codes.shape) == 1:
        weight = codes
    else:
        weight = codes[:, dim]
    weight_indices = np.argsort(-weight)
    n = int(len(weight_indices))
    n_rows = 7
    n_cols = 12

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 15))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.0)

    ws = 12  # window size

    for i, p in enumerate(np.arange(0, n_rows * 2, 2)):
        middle_idx = int(n * ((p / 100)))
        start_idx = max(0, middle_idx - ws // 2)  # for the lowest 20 percentile
        end_idx = min(n, middle_idx + ws // 2)  # for the highest 20 percentile
        end_idx = max(end_idx, ws)
        indices = list(range(start_idx, end_idx))
        # random.seed(0)
        # random.shuffle(indices)
        indices = indices[:12]
        weight_percentile = weight_indices[indices]
        for j, sample in enumerate(weight_percentile):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
            ax.set_ylabel(str(p))
            img = io.imread(images[sample])
            ax.imshow(img)

            if j == 11:
                break
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

    n_dims = W.shape[1]

    for dim in range(n_dims):
        if args.behav_experiment:
            behav_path = os.path.join(
                base_path, "analyses", "behavior_experiment", "images"
            )
            if not os.path.exists(behav_path):
                os.makedirs(behav_path)
            fig_large = plot_behavior(images, W, dim, top_k=16)
            for ext in ["jpg"]:
                fname = os.path.join(behav_path, f"{dim}_topk{append}_large.{ext}")
                fig_large.savefig(fname, dpi=100, bbox_inches="tight", pad_inches=0)
            plt.close(fig_large)
            print(f"Done plotting for dim {dim} large")
            continue

        fig_5x2 = plot_dim(images, W, dim, 10)
        fig_3x3 = plot_dim_3x3(images, W, dim, 9)
        fig_1x8 = plot_dim_1x8(images, W, dim, 8)
        fig_5x10 = plot_top_50(images, W, dim, 50)

        # fig.suptitle("Dimension: {}".format(dim))
        out_path = os.path.join(results_path, f"{dim:02d}")
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        for ext in ["png", "pdf"]:
            fname = os.path.join(out_path, f"{dim}_topk{append}_5x2.{ext}")
            fig_5x2.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0)
            fname = os.path.join(out_path, f"{dim}_topk{append}_3x3.{ext}")
            fig_3x3.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0)
            fname = os.path.join(out_path, f"{dim}_topk{append}_1x8.{ext}")
            fig_1x8.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0)
            fname = os.path.join(out_path, f"{dim}_topk{append}_5x10.{ext}")
            fig_5x10.savefig(fname, dpi=150, bbox_inches="tight", pad_inches=0)

        plt.close(fig_5x2)
        plt.close(fig_3x3)
        plt.close(fig_1x8)
        plt.close(fig_5x10)
        print(f"Done plotting for dim {dim}")


def plot_dimensions(args):
    W = load_sparse_codes(args.embedding_path)

    images, indices = load_image_data(
        args.img_root,
        filter_behavior=args.filter_behavior,
        filter_plus=args.filter_plus,
    )
    W = W[indices]

    print("Shape of weight Matrix", W.shape)
    W = W.T

    base_path = os.path.dirname(os.path.dirname(args.embedding_path))
    filename = os.path.basename(args.embedding_path)
    epoch = filename.split("_")[-1].split(".")[0]
    out_path = os.path.join(base_path, "analyses")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Plot W as a matrix with dots in black and white
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(W, cmap="gray", interpolation="nearest")
    ax.axis("off")
    fig.savefig(
        os.path.join(out_path, "weight_matrix.pdf"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)

    top_k = 20
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

    fname = os.path.join(out_path, "all_dimensions{}_epoch_{}.png")

    if args.filter_behavior:
        fname = fname.format("_filtered_behavior", epoch)
    if args.filter_plus:
        fname = fname.format("_filtered_plus", epoch)
    else:
        fname = fname.format("", epoch)
    fig.savefig(fname, dpi=50, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    args = parse_args()
    if args.per_dim:
        plot_per_dim(args)
    else:
        plot_dimensions(args)
