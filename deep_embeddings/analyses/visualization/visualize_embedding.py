#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np

from deep_embeddings.utils.utils import load_sparse_codes, load_image_data, filter_embedding_by_behavior

parser = argparse.ArgumentParser(description='Visualize embedding')
parser.add_argument('--embedding_path', type=str, default='./weights', help='path to weights directory') # TODO remove default
parser.add_argument('--n_images', type=int, default=12, choices=[6,12], help='number of images per category')
parser.add_argument('--modality', type=str, default="deep", choices=("behavior", "deep"), help='if behavior images or not')
parser.add_argument('--filter_images',  type=str, default="False", choices=("True", "False"), help="If to filter the learned embedding by the behavior images")
parser.add_argument('--per_dim', type=str, choices=("True", "False"), help="Plots per dimension if true")


def plot_per_dim(args):
    base_path = os.path.dirname(os.path.dirname(args.embedding_path))
    results_path = os.path.join(base_path, "analyses", "per_dim")
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    W = load_sparse_codes(args.embedding_path)
    _, _, images = load_image_data(args.n_images, args.modality)
    print('Shape of weight Matrix', W.shape)
    W = W.T

    top_k = 16
    top_j = W.shape[1]

    for dim, w_j in enumerate(W):
        fig = plt.figure(figsize=(4,4))
        gs1 = gridspec.GridSpec(4,4)
        gs1.update(wspace=0.002, hspace=0.002) # set the spacing between axes.
        top_k_samples = np.argsort(-w_j)[:top_k] # this is over the image dimension
        for k, sample in enumerate(top_k_samples):
            ax = plt.subplot(gs1[k])
            ax.imshow(io.imread(images[sample]))
            ax.set_xticks([])
            ax.set_yticks([])
    
        # fig.suptitle("Dimension: {}".format(dim))
        out_path = os.path.join(results_path, f'{dim:02d}')
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        fname = os.path.join(out_path, f'dim_{dim}_topk.png')
        fig.savefig(fname, dpi=300)
        plt.close(fig)
        print(f'Done plotting for dim {dim}')

        if dim > top_j:
            break    

def plot_dimensions(args):
    _, _, images = load_image_data(args.n_images, args.modality)
    W = load_sparse_codes(args.embedding_path)    
    if args.filter_images == "True":
        print("Select only behavior images to visualize the embedding")
        W, images = filter_embedding_by_behavior(W, images)

    print('Shape of weight Matrix', W.shape)
    W = W.T

    top_j = 60
    top_k = 12

    n_rows = top_j if top_j <= W.shape[1] else W.shape[1]
    n_cols = top_k
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 120))

    for j, w_j in enumerate(W):
        top_k_samples = np.argsort(-w_j)[:top_k] # this is over the image dimension
        for k, sample in enumerate(top_k_samples):
            axes[j, k].imshow(io.imread(images[sample]))
            # axes[j, k].set_xlabel(f'Weight: {w_j[sample]:.2f}', fontsize=20)
            axes[j, k].set_xticks([])
            axes[j, k].set_yticks([])

        if j == n_rows-1:
            break

    plt.tight_layout()

    # Save figure in predefined embedding path directory
    base_path = os.path.dirname(os.path.dirname(args.embedding_path))
    filename = os.path.basename(args.embedding_path)
    epoch = filename.split('_')[-1].split('.')[0]
    out_path = os.path.join(base_path, "analyses")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    fname = os.path.join(out_path, "all_dimensions{}_epoch_{}.png")

    if args.filter_images == "True":
        fname = fname.format("_filtered", epoch)
    else:
        fname = fname.format("", epoch)
    fig.savefig(fname, dpi=50)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.filter_images == "True" and args.modality == 'behavior':
        raise ValueError("Cannot filter behavior images for behavior images")

    if args.per_dim == "True":    
        plot_per_dim(args)
    else:
        plot_dimensions(args)
