#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.transform import resize
import skimage.io as io

from deep_embeddings.utils.utils import load_sparse_codes, filter_embedding_by_behavior, fill_diag, correlate_rsms, correlation_matrix
from scipy.stats import rankdata, pearsonr, spearmanr

import matplotlib.gridspec as gridspec
import seaborn as sns


parser = argparse.ArgumentParser(description='Compare human and DNN performance on the same task.')
parser.add_argument('--human_embedding_path', type=str, help='Path to human embedding matrix.')
parser.add_argument('--dnn_embedding_path', type=str, help='Path to DNN embedding matrix.')
parser.add_argument('--feature_path', type=str, help='Path to VGG feature matrix and filenames')


def get_img_pairs(tril_indices, most_dissimilar):
    """ Returns the most dissimlar image pair indices in the reference images from the lower triangular RSM matrix"""
    tril_inds_i= tril_indices[0][most_dissimilar]
    tril_inds_j = tril_indices[1][most_dissimilar]
    img_pair_indices = np.array([(i,j) for i, j in zip(tril_inds_i, tril_inds_j)])

    return img_pair_indices


def compare_modalities(weights_human, weights_dnn, duplicates=False):
    """ Compare the Human behavior embedding to the VGG embedding by correlating them! """
    assert weights_dnn.shape[0] == weights_human.shape[0], '\nNumber of items in weight matrices must align.\n'

    dim_human = weights_human.shape[1]
    dim_dnn = weights_dnn.shape[1]

    if dim_human < dim_dnn:
        dim_smaller = dim_human
        dim_larger = dim_dnn
        mod_1 = weights_human
        mod_2 = weights_dnn    

    else:
        dim_smaller = dim_dnn
        dim_larger = dim_human
        mod_1 = weights_dnn
        mod_2 = weights_human

    corrs_between_modalities = np.zeros(dim_smaller)
    mod2_dims = []

    for dim_idx_1, weight_1 in enumerate(mod_1.T):
        # Correlate modality 1 with all dimensions of modality 2
        corrs = np.zeros(dim_larger)
        for dim_idx_2, weight_2 in enumerate(mod_2.T):
            corrs[dim_idx_2] = pearsonr(weight_1, weight_2)[0]

        if duplicates:
            mod2_dims.append(np.argmax(corrs))

        else:
            for corrs_mod_2 in np.argsort(-corrs):
                if corrs_mod_2 not in mod2_dims:
                    mod2_dims.append(corrs_mod_2)
                    break

        corrs_between_modalities[dim_idx_1] = corrs[mod2_dims[-1]]

    # Sort the dimensions based on highest correlations!
    mod1_dims_sorted = np.argsort(-corrs_between_modalities)
    mod2_dims_sorted = np.asarray(mod2_dims)[mod1_dims_sorted]
    corrs = corrs_between_modalities[mod1_dims_sorted]

    return mod1_dims_sorted, mod2_dims_sorted, corrs
            


def plot_most_dissim_pairs(plots_dir, ref_images, weights_human, weights_dnn, top_k=4):
    #create directory for density scatter plots
    
    path = os.path.join(plots_dir, 'per_dim')

    if not os.path.exists(path):
        print(f'\nCreating directories...\n')
        os.makedirs(path)

    weights_human = weights_human.T
    weights_dnn = weights_dnn.T

    n_dims = weights_human.shape[0]

    #plot most dissimilar pairs per dimension
    for dim_idx in range(n_dims):

        h_dim = weights_human[dim_idx]
        d_dim = weights_dnn[dim_idx]


        most_sim_h = np.argsort(-h_dim)[:top_k]
        most_sim_d = np.argsort(-d_dim)[:top_k]

        most_dissim_h = np.argsort(h_dim)[:top_k]
        most_dissim_d = np.argsort(d_dim)[:top_k]

        colors = ['red', 'blue']
        human_imgs, dnn_imgs = [], []
        fig, axes = plt.subplots(2,2)
    
        for (sim_h, sim_d) in zip(most_sim_h, most_sim_d):
            path_i = ref_images[sim_h]
            path_j = ref_images[sim_d]
            img_i = resize(io.imread(path_i), (400,400))
            img_j = resize(io.imread(path_j), (400,400))
            human_imgs.append(img_i)
            dnn_imgs.append(img_j)


        human_imgs_row = np.concatenate(human_imgs[0:2], axis=1)
        human_imgs_col = np.concatenate(human_imgs[2:4], axis=1)
        human_imgs = np.concatenate((human_imgs_row, human_imgs_col), axis=0)
        dnn_imgs_row = np.concatenate(dnn_imgs[0:2], axis=1)
        dnn_imgs_col = np.concatenate(dnn_imgs[2:4], axis=1)
        dnn_imgs = np.concatenate((dnn_imgs_row, dnn_imgs_col), axis=0)

        axes[0, 0].imshow(human_imgs)
        axes[0, 1].imshow(dnn_imgs)

        axes[0, 0].set_title('Human')
        axes[0, 1].set_title('DNN')

        # for i in range(2):
        #     for spine in axes[i].spines:
        #         axes[i].spines[spine].set_color(colors[i])
        #         axes[i].spines[spine].set_linewidth(1.6)
        #         axes[i].set_xticks([])
        #         axes[i].set_yticks([])

        
        # fname = os.path.join(path, f'high_human_low_dnn_{dim_idx}.png')
        # fig.savefig(fname, dpi=300, bbox_inches='tight')
        # plt.close()

        # colors = ['red', 'blue']
        human_imgs, dnn_imgs = [], []
        # fig, axes = plt.subplots(1,2)
        for (sim_h, sim_d) in zip(most_dissim_h, most_dissim_d):
            path_i = ref_images[sim_h]
            path_j = ref_images[sim_d]
            img_i = resize(io.imread(path_i), (400,400))
            img_j = resize(io.imread(path_j), (400,400))
            human_imgs.append(img_i)
            dnn_imgs.append(img_j)

        human_imgs_row = np.concatenate(human_imgs[0:2], axis=1)
        human_imgs_col = np.concatenate(human_imgs[2:4], axis=1)
        human_imgs = np.concatenate((human_imgs_row, human_imgs_col), axis=0)
        dnn_imgs_row = np.concatenate(dnn_imgs[0:2], axis=1)
        dnn_imgs_col = np.concatenate(dnn_imgs[2:4], axis=1)
        dnn_imgs = np.concatenate((dnn_imgs_row, dnn_imgs_col), axis=0)

        axes[1, 0].imshow(human_imgs)
        axes[1, 1].imshow(dnn_imgs)

        # axes[0].set_title('Human')
        # axes[1].set_title('DNN')
        axes[0,0].set_ylabel('Most similar')
        axes[1,0].set_ylabel('Most dissimilar')

        for ax in axes.reshape(-1):
            # # for spine in axes[i].spines:        
            ax.set_xticks([])
            ax.set_yticks([])

    
        plt.subplots_adjust(hspace=0.1, wspace=-0.05)
        fname = os.path.join(path, f'comparison_dim_{dim_idx}.png')
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()


def compare_human_dnn(human_embedding_path, dnn_embedding_path, feature_path):
    """Compare human and DNN performance on the same task."""

    weights_dnn = load_sparse_codes(dnn_embedding_path)
    weights_human = load_sparse_codes(human_embedding_path)

    filenames_path = os.path.join(feature_path, 'file_names.txt')
    if not os.path.exists(filenames_path):
        raise FileNotFoundError("File names not found in DNN activation path {}".format(feature_path))

    filenames = np.loadtxt(filenames_path, dtype=str)

    # Filter out images without behavioral data
    weights_dnn, ref_images = filter_embedding_by_behavior(weights_dnn, filenames)

    # Get rsm matrices
    rsm_dnn = correlation_matrix(weights_dnn)
    rsm_human = correlation_matrix(weights_human)

    rho = correlate_rsms(rsm_dnn, rsm_human, "correlation")
    print("Correlation between human and DNN embeddings: {}".format(rho))


    plot_dir_comparison = os.path.dirname(os.path.dirname(dnn_embedding_path))
    plot_dir_comparison = os.path.join(plot_dir_comparison, "analyses", "human_dnn")
    if not os.path.exists(plot_dir_comparison):
        os.makedirs(plot_dir_comparison)


    # Plot mind machine correlations in decreasing order!
    fig, ax = plt.subplots()
    human_sorted_indices, dnn_sorted_indices, mind_machine_corrs = compare_modalities(weights_human, weights_dnn, duplicates=True)
    sns.lineplot(x=range(len(mind_machine_corrs)), y=mind_machine_corrs, ax=ax)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Correlation")
    ax.set_title("Human vs DNN dimensions")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir_comparison, "human_dnn_dimension_correlation.png"), dpi=300)


    print("Plot most dissimilar object pairs")
    weights_human_sorted = weights_human[:, human_sorted_indices]
    weights_dnn_sorted = weights_dnn[:, dnn_sorted_indices]
    plot_most_dissim_pairs(plot_dir_comparison, ref_images, weights_human_sorted, weights_dnn_sorted)



if __name__ == '__main__':
    args = parser.parse_args()
    compare_human_dnn(args.human_embedding_path, args.dnn_embedding_path, args.feature_path)






    



