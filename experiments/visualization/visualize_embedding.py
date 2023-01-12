#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main visualization of embedding dimension based on sorted weights"""

import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np

from deep_embeddings import ExperimentParser
from deep_embeddings.utils.utils import load_sparse_codes, load_image_data


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


def plot_per_dim(args):
    base_path = os.path.dirname(os.path.dirname(args.embedding_path))
    results_path = os.path.join(base_path, "analyses", "per_dim")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    W = load_sparse_codes(args.embedding_path)
    images, indices = load_image_data(args.img_root, filter_behavior=args.filter_behavior, filter_plus=args.filter_plus)
    
    if W.shape[0] != len(images):
        W = W[indices]

    print("Shape of weight Matrix", W.shape)
    W = W.T

    top_k = 10
    top_j = W.shape[1]

    for dim, w_j in enumerate(W):
        fig = plt.figure(figsize=(5,2))
        gs1 = gridspec.GridSpec(2, 5)
        gs1.update(wspace=0.0, hspace=0.0)  # set the spacing between axes.
        top_k_samples = np.argsort(-w_j)[:top_k]  # this is over the image dimension
        for k, sample in enumerate(top_k_samples):
            ax = plt.subplot(gs1[k])
            ax.imshow(io.imread(images[sample]))
            ax.set_xticks([])
            ax.set_yticks([])

        # fig.suptitle("Dimension: {}".format(dim))
        out_path = os.path.join(results_path, f"{dim:02d}")
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        fname = os.path.join(out_path, f"dim_{dim}_topk.png")
        fig.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0)
    
        plt.close(fig)

        print(f"Done plotting for dim {dim}")

        if dim > top_j:
            break


def plot_dimensions(args):
    W = load_sparse_codes(args.embedding_path)
    images, indices = load_image_data(args.img_root, filter_behavior=args.filter_behavior, filter_plus=args.filter_plus)
    
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
    fig.savefig(fname, dpi=50)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.per_dim:
        plot_per_dim(args)
    else:
        plot_dimensions(args)
