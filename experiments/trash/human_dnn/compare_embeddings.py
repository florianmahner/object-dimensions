#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from skimage.transform import resize
import skimage.io as io

from object_dimensions.utils.utils import (
    load_sparse_codes,
    filter_embedding_by_behavior,
    fill_diag,
    correlate_rsms,
    correlation_matrix,
)
from scipy.stats import rankdata, pearsonr, spearmanr

import matplotlib.gridspec as gridspec
import seaborn as sns


parser = argparse.ArgumentParser(
    description="Compare human and DNN performance on the same task."
)
parser.add_argument(
    "--human_embedding_path", type=str, help="Path to human embedding matrix."
)
parser.add_argument(
    "--dnn_embedding_path", type=str, help="Path to DNN embedding matrix."
)
parser.add_argument(
    "--feature_path", type=str, help="Path to VGG feature matrix and filenames"
)


def get_img_pairs(tril_indices, most_dissimilar):
    """Returns the most dissimlar image pair indices in the reference images from the lower triangular RSM matrix"""
    tril_inds_i = tril_indices[0][most_dissimilar]
    tril_inds_j = tril_indices[1][most_dissimilar]
    img_pair_indices = np.array([(i, j) for i, j in zip(tril_inds_i, tril_inds_j)])

    return img_pair_indices


def compare_modalities(weights_human, weights_dnn, duplicates=False):
    """Compare the Human behavior embedding to the VGG embedding by correlating them!"""
    assert (
        weights_dnn.shape[0] == weights_human.shape[0]
    ), "\nNumber of items in weight matrices must align.\n"

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
    """This function plots the most dissimilar image pairs between human and dnn representations based
    on the largest difference in weight vectors within a given dimension"""

    path = os.path.join(plots_dir, "per_dim")

    if not os.path.exists(path):
        print(f"\nCreating directories...\n")
        os.makedirs(path)

    weights_human = weights_human.T
    weights_dnn = weights_dnn.T

    n_dims = weights_human.shape[0]

    # plot most dissimilar pairs per dimension
    for dim_idx in range(n_dims):

        h_dim = weights_human[dim_idx]
        d_dim = weights_dnn[dim_idx]

        most_sim_h = np.argsort(-h_dim)[:top_k]
        most_sim_d = np.argsort(-d_dim)[:top_k]

        most_dissim_h = np.argsort(h_dim)[:top_k]
        most_dissim_d = np.argsort(d_dim)[:top_k]

        colors = ["red", "blue"]
        human_imgs, dnn_imgs = [], []
        fig, axes = plt.subplots(2, 2)

        for (sim_h, sim_d) in zip(most_sim_h, most_sim_d):
            path_i = ref_images[sim_h]
            path_j = ref_images[sim_d]
            img_i = resize(io.imread(path_i), (400, 400))
            img_j = resize(io.imread(path_j), (400, 400))
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

        axes[0, 0].set_title("Human")
        axes[0, 1].set_title("DNN")

        human_imgs, dnn_imgs = [], []
        for (sim_h, sim_d) in zip(most_dissim_h, most_dissim_d):
            path_i = ref_images[sim_h]
            path_j = ref_images[sim_d]
            img_i = resize(io.imread(path_i), (400, 400))
            img_j = resize(io.imread(path_j), (400, 400))
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

        axes[0, 0].set_ylabel("Most similar")
        axes[1, 0].set_ylabel("Most dissimilar")

        for ax in axes.reshape(-1):
            ax.set_xticks([])
            ax.set_yticks([])

        plt.subplots_adjust(hspace=0.1, wspace=-0.05)
        fname = os.path.join(path, f"comparison_dim_{dim_idx}.png")
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()


def plot_most_dissim_dims(
    plots_dir, human_dnn_corrs, ref_images, weights_human, weights_dnn, top_k=4
):
    """This function plots the most dissimilar image pairs between human and dnn representations based
    on the largest difference in weight vectors within a given dimension"""

    weights_human = weights_human.T
    weights_dnn = weights_dnn.T

    n_dims = weights_human.shape[0]
    n_objects = weights_human.shape[1]
    topk = 50

    # plot most dissimilar pairs per dimension
    for dim_idx in tqdm.tqdm(range(n_dims)):
        # Include leading zeros in the dimension index
        dim_idx_str = str(dim_idx).zfill(2)
        path = os.path.join(plots_dir, dim_idx_str)
        if not os.path.exists(path):
            os.makedirs(path)

        human_dim = weights_human[dim_idx]
        dnn_dim = weights_dnn[dim_idx]

        rank_human = rankdata(human_dim)
        rank_dnn = rankdata(dnn_dim)

        high_both = np.mean([rank_human, rank_dnn], axis=0)
        high_both = np.argsort(-high_both)[:topk]

        high_human_low_dnn = np.mean([rank_human, n_objects - rank_dnn], axis=0)
        high_human_low_dnn = np.argsort(-high_human_low_dnn)[:topk]

        high_dnn_low_human = np.mean([rank_dnn, n_objects - rank_human], axis=0)
        high_dnn_low_human = np.argsort(-high_dnn_low_human)[:topk]

        low_both = np.mean([n_objects - rank_human, n_objects - rank_dnn], axis=0)
        low_both = np.argsort(-low_both)[:topk]

        for iden, img_indices in zip(
            [
                "High Human High DNN",
                "High Human Low Dnn",
                "High DNN Low Human",
                "Low Human Low DNN",
            ],
            [high_both, high_human_low_dnn, high_dnn_low_human, low_both],
        ):

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 4))

            path_i = ref_images[img_indices[0]]
            path_j = ref_images[img_indices[1]]
            img_i = resize(io.imread(path_i), (400, 400))
            img_j = resize(io.imread(path_j), (400, 400))
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.set_xticks([])
            ax2.set_yticks([])

            # Make a red border around ax1 and a blue border around ax2 with the spines
            for spine in ax1.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(1.5)

            for spine in ax2.spines.values():
                spine.set_edgecolor("blue")
                spine.set_linewidth(1.5)

            ax1.imshow(img_i)
            ax2.imshow(img_j)
            # plt.suptitle(iden)

            iden = iden.replace(" ", "_")
            fname = os.path.join(path, f"{iden}.png")
            # fig.tight_layout(rect=[0, 0.6, 1.0, 0.95])
            fig.tight_layout()

            fig.savefig(fname, dpi=300, bbox_inches="tight")
            fig.close()

        fig, ax = plt.subplots(1, 1)
        plot_dict = {}
        plot_dict["human"] = human_dim
        plot_dict["dnn"] = dnn_dim
        plot_dict["cluster"] = ["silver"] * len(human_dim)

        # We want to highlight the values where humans are high and dnn are low in red and where dnns are high and humans are low in blue
        for idx_1, idx_2 in zip(high_human_low_dnn, high_dnn_low_human):
            plot_dict["cluster"][idx_1] = "red"
            plot_dict["cluster"][idx_2] = "blue"

        # Plot the cluster assignments as joint plot
        plot_dict = pd.DataFrame(plot_dict)

        with sns.axes_style("white"):
            x = human_dim
            y = dnn_dim

            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            g = sns.jointplot(
                data=plot_dict,
                x="human",
                y="dnn",
                hue="cluster",
                legend=False,
                ax=ax,
                s=25,
            )

            m, b = np.polyfit(x, y, deg=1)

            # Draw (regression) line of best fit
            g.ax_joint.plot(x, m * x + b, linewidth=2, c="black")
            g.ax_joint.set_xticks([])
            g.ax_joint.set_yticks([])
            g.ax_joint.set_xlabel("Human", fontsize=13)
            g.ax_joint.set_ylabel("DNN", fontsize=13)

            # Annotate correlation coefficient
            rho = human_dnn_corrs[dim_idx]
            corr_str = r"$\rho$" + " = " + str(rho)
            loc_x = np.min(x)
            loc_y = np.max(y)
            g.ax_joint.annotate(corr_str, (loc_x, loc_y), fontsize=11)
            plt.title("Dimension {}".format(dim_idx), fontsize=12)

            fname = os.path.join(path, f"scatter.png")
            # plt.savefig(fname, dpi=150, bbox_inches='tight')
            g.savefig(fname, bbox_inches="tight", dpi=300)
            fig.close()


def find_rank_transformed_dissimilarities(
    weights_human, weights_dnn, ref_images, plots_dir, topk=4
):

    n_objects = weights_human.shape[0]
    n_dim = min(weights_human.shape[1], weights_dnn.shape[1])

    for dim_idx in range(n_dim):

        dim_path = os.path.join(plots_dir, str(dim_idx).zfill(2))
        if not os.path.exists(dim_path):
            os.makedirs(dim_path)

        print("Processing dimension {}".format(dim_idx))

        rank_human = rankdata(weights_human[:, dim_idx])
        rank_dnn = rankdata(weights_dnn[:, dim_idx])

        high_both = np.mean([rank_human, rank_dnn], axis=0)
        high_both = np.argsort(-high_both)[:topk]

        high_human_low_dnn = np.mean([rank_human, n_objects - rank_dnn], axis=0)
        high_human_low_dnn = np.argsort(-high_human_low_dnn)[:topk]

        high_dnn_low_human = np.mean([rank_dnn, n_objects - rank_human], axis=0)
        high_dnn_low_human = np.argsort(-high_dnn_low_human)[:topk]

        for (img_indices, identifier) in zip(
            [high_both, high_human_low_dnn, high_dnn_low_human],
            ["High Human High DNN", "High Human Low Dnn", "High DNN Low Human"],
        ):

            fig, axes = plt.subplots(1, topk)

            for i in range(topk):
                path_i = ref_images[img_indices[i]]
                img_i = resize(io.imread(path_i), (400, 400))
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                axes[i].imshow(img_i)

            identifier = identifier.replace(" ", "_")
            fname = os.path.join(dim_path, f"{identifier}.png")
            fig.tight_layout()
            fig.savefig(fname, dpi=300, bbox_inches="tight")
            plt.close()


def compare_human_dnn(human_embedding_path, dnn_embedding_path, feature_path):
    """Compare human and DNN performance on the same task."""
    weights_dnn = load_sparse_codes(dnn_embedding_path)
    weights_human = load_sparse_codes(human_embedding_path)

    filenames_path = os.path.join(feature_path, "filenames.txt")
    if not os.path.exists(filenames_path):
        raise FileNotFoundError(
            "File names not found in DNN activation path {}".format(feature_path)
        )

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
    human_sorted_indices, dnn_sorted_indices, mind_machine_corrs = compare_modalities(
        weights_human, weights_dnn, duplicates=True
    )
    sns.lineplot(x=range(len(mind_machine_corrs)), y=mind_machine_corrs, ax=ax)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Correlation")
    ax.set_title("Human vs DNN dimensions")
    fig.tight_layout()
    fig.savefig(
        os.path.join(plot_dir_comparison, "human_dnn_dimension_correlation.png"),
        dpi=300,
    )

    print(plot_dir_comparison)

    path = os.path.join(plot_dir_comparison, "per_dim")
    if not os.path.exists(path):
        print(f"\nCreating directories...\n")
        os.makedirs(path)

    print("Plot most dissimilar object pairs")
    weights_human_sorted = weights_human[:, human_sorted_indices]
    weights_dnn_sorted = weights_dnn[:, dnn_sorted_indices]

    find_rank_transformed_dissimilarities(
        weights_human_sorted, weights_dnn_sorted, ref_images, path, topk=4
    )

    # plot_most_dissim_dims(plot_dir_comparison, mind_machine_corrs, ref_images, weights_human_sorted, weights_dnn_sorted)


if __name__ == "__main__":
    args = parser.parse_args()
    compare_human_dnn(
        args.human_embedding_path, args.dnn_embedding_path, args.feature_path
    )
