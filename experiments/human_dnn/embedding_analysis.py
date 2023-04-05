#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.transform import resize
import skimage.io as io

from object_dimensions.utils.utils import correlate_rsms, correlation_matrix
from scipy.stats import rankdata, pearsonr


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


import numpy as np
from scipy.stats import pearsonr

def compare_modalities(weights_human, weights_dnn, duplicates=False, not_sorted=False, all_dims=False):
    """Compares the Human behavior embedding to the VGG embedding by correlating them."""
    
    assert weights_dnn.shape[0] == weights_human.shape[0], "Number of items in weight matrices must align."
    dim_human, dim_dnn = weights_human.shape[1], weights_dnn.shape[1]
    if dim_human < dim_dnn:
        dim_smaller, dim_larger = dim_human, dim_dnn
        mod_1, mod_2 = weights_human, weights_dnn
    else:
        dim_smaller, dim_larger = dim_dnn, dim_human
        mod_1, mod_2 = weights_dnn, weights_human

    corrs_between_modalities = np.zeros(dim_smaller)
    mod2_dims = []
    corrs_for_all_dims = []
    for dim_idx_1, weight_1 in enumerate(mod_1.T):
        corrs = np.zeros(dim_larger)
        for dim_idx_2, weight_2 in enumerate(mod_2.T):
            corrs[dim_idx_2] = pearsonr(weight_1, weight_2)[0]
        
        corrs_for_all_dims.append([corrs])
        
        if duplicates:
            mod2_dims.append(np.argmax(corrs))
        # If duplicates are not allowed, take the highest correlation that has not been used before
        # i.e. is not in mod2_dims
        else:
            for corrs_mod_2 in np.argsort(-corrs):
                if corrs_mod_2 not in mod2_dims:
                    mod2_dims.append(corrs_mod_2)
                    break

        # Store the highest correlation for the current dimension
        corrs_between_modalities[dim_idx_1] = corrs[mod2_dims[-1]]

    # Sort the dimensions based on the highest correlations
    mod1_dims_sorted = np.argsort(-corrs_between_modalities)
    mod2_dims_sorted = np.asarray(mod2_dims)[mod1_dims_sorted]
    corrs = corrs_between_modalities[mod1_dims_sorted]

    corrs_for_all_dims = np.concatenate(corrs_for_all_dims, axis=0)

    # Return the sorted dimensions and correlations
    if not_sorted:
        if all_dims:
            return mod1_dims_sorted, mod2_dims_sorted, corrs_between_modalities, corrs_for_all_dims
        return mod1_dims_sorted, mod2_dims_sorted, corrs_between_modalities
    elif all_dims:
        return mod1_dims_sorted, mod2_dims_sorted, corrs, corrs_for_all_dims
    else:
        return 



def get_img_pairs(tril_indices, most_dissimilar):
    """Returns the most dissimlar image pair indices in the reference images from the lower triangular RSM matrix"""
    tril_inds_i = tril_indices[0][most_dissimilar]
    tril_inds_j = tril_indices[1][most_dissimilar]
    img_pair_indices = np.array([(i, j) for i, j in zip(tril_inds_i, tril_inds_j)])

    return img_pair_indices


def plot_density_scatters(
    plots_dir,
    behavior_images,
    rsm_1,
    rsm_2,
    concepts,
    top_k,
    mod_1="Human",
    mod_2="VGG16",
):
    assert rsm_1.shape == rsm_2.shape, "\nRSMs must be of equal size.\n"

    scatter_dir = "density_scatters"
    path = os.path.join(plots_dir, scatter_dir)
    if not os.path.exists(path):
        print(f"\nCreating directories...\n")
        os.makedirs(path)

    colors = ["silver", "red", "blue"]
    for i, concept in enumerate(concepts.columns):
        # Find indices in the concept matrix where the concept is present
        objects_with_concept = np.where(concepts.loc[:, concept] == 1)[0]

        # Extract the RSMs for the concept (i.e. the rows and columns corresponding to the concept)
        rsm_1_concept = rsm_1[objects_with_concept]
        rsm_1_concept = rsm_1_concept[:, objects_with_concept]
        rsm_2_concept = rsm_2[objects_with_concept]
        rsm_2_concept = rsm_2_concept[:, objects_with_concept]

        # Compute lower triangular parts of symmetric RSMs (zero elements above and including main diagonal) filtered for objects with concept
        tril_inds = np.tril_indices(len(rsm_1_concept), k=-1)
        tril_1 = rsm_1_concept[tril_inds]
        tril_2 = rsm_2_concept[tril_inds]
        rho = pearsonr(tril_1, tril_2)[0].round(5)
        # rho = spearmanr(tril_1, tril_2)[0].round(5)

        # Find object pairs that are most dissimilar between modality one (i.e. behavior) and modality 2 (i.e. VGG 16) for a specific concept
        most_dissim_1 = np.argsort(rankdata(tril_1) - rankdata(tril_2))[::-1][:top_k]
        most_dissim_2 = np.argsort(rankdata(tril_2) - rankdata(tril_1))[::-1][:top_k]
        most_dissim_pairs = (most_dissim_1, most_dissim_2)

        # Store thes indices in a 1d array of all lower triangular indices with most dissimilar objects
        categories = np.zeros(len(tril_1), dtype=int)
        categories[most_dissim_1] += 1
        categories[most_dissim_2] += 2

        # Store the lower triangular RSM values for both modalities along side the category cluster assignment for a single concept
        object_similarities = pd.DataFrame(
            np.c_[tril_1, tril_2, categories], columns=[mod_1, mod_2, "category"]
        )

        with sns.axes_style("white"):
            g = sns.jointplot(
                data=object_similarities,
                x=mod_1,
                y=mod_2,
                kind="scatter",
                hue="category",
                palette=dict(enumerate(colors)),
                legend=False,
                height=7,
            )
            x = object_similarities[mod_1]
            y = object_similarities[mod_2]
            m, b = np.polyfit(x, y, deg=1)

            # Draw (regression) line of best fit
            g.ax_joint.plot(x, m * x + b, linewidth=2, c="black")
            g.ax_joint.set_xticks([])
            g.ax_joint.set_yticks([])
            g.ax_joint.set_xlabel(mod_1, fontsize=13)
            g.ax_joint.set_ylabel(mod_2, fontsize=13)

            # Annotate correlation coefficient
            corr_str = r"$\rho$" + " = " + str(rho)
            loc_x = np.min(tril_1)
            loc_y = np.max(tril_2)
            g.ax_joint.annotate(corr_str, (loc_x, loc_y), fontsize=11)
            plt.title(concept)

            # Save figure
            concept_path = os.path.join(path, "concepts")
            if not os.path.exists(concept_path):
                os.makedirs(concept_path)
            fname = os.path.join(concept_path, f"{concept}.png")
            g.savefig(fname, bbox_inches="tight", dpi=300)
            plt.close()

        # Visualize the most dissimilar object pairs as images!
        mods = [mod_1, mod_2]
        behavior_images_c = behavior_images[objects_with_concept]

        fig, axes = plt.subplots(1, 2)

        for i, most_dissim in enumerate(most_dissim_pairs):
            img_pairs = get_img_pairs(tril_inds, most_dissim)

            for spine in axes[i].spines:
                axes[i].spines[spine].set_color(colors[i + 1])
                axes[i].spines[spine].set_linewidth(1.6)

            pair_1, pair_2 = img_pairs[i]
            path_1 = behavior_images_c[pair_1]
            path_2 = behavior_images_c[pair_2]

            img_1 = resize(io.imread(path_1), (400, 400))
            img_2 = resize(io.imread(path_2), (400, 400))
            img = np.concatenate((img_1, img_2), axis=1)
            axes[i].imshow(img)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_title(mods[i], fontsize=10)

        # mod_str = mods[i].lower().split()
        # mod_str = '_'.join(mod_str)
        dissim_path = os.path.join(path, "most_dissim_obj_pairs")
        if not os.path.exists(dissim_path):
            os.makedirs(dissim_path)

        fname = os.path.join(dissim_path, f"{concept}.png")
        fig.tight_layout()
        fig.savefig(fname, bbox_inches="tight", dpi=300)
        plt.close()


def plot_most_dissim_pairs(plots_dir, rsm_1, rsm_2, mod_1, mod_2, top_k):
    # create directory for density scatter plots
    PATH = os.path.join(plots_dir, "".join(("density_scatters", "_", "overall")))
    if not os.path.exists(PATH):
        print(f"\nCreating directories...\n")
        os.makedirs(PATH)

    tril_inds = np.tril_indices(len(rsm_1), k=-1)

    tril_1 = rsm_1[tril_inds]
    tril_2 = rsm_2[tril_inds]

    # Compute pearson correlation of lower triangular parts of RSMs
    rho = pearsonr(tril_1, tril_2)[0].round(3)

    # Subtract rank of each element in tril_mod1 from rank of each element in tril_mod2
    most_dissim_1 = np.argsort(tril_1 - tril_2)[::-1][:top_k]
    most_dissim_2 = np.argsort(tril_2 - tril_1)[::-1][:top_k]
    most_dissim_pairs = [most_dissim_1, most_dissim_2]
    colors = ["grey", "red", "blue"]

    labels = np.zeros(len(tril_1))
    labels[most_dissim_1] += 1
    labels[most_dissim_2] += 2
    obj_sims = pd.DataFrame(
        np.c_[tril_1, tril_2, labels], columns=[mod_1, mod_2, "labels"]
    )
    with sns.axes_style("ticks"):
        g = sns.jointplot(
            data=obj_sims,
            x=mod_1,
            y=mod_2,
            hue="labels",
            palette=dict(enumerate(colors)),
            height=7,
            alpha=0.6,
            kind="scatter",
            legend=False,
        )
        x = obj_sims[mod_1]
        y = obj_sims[mod_2]
        m, b = np.polyfit(x, y, 1)
        g.ax_joint.plot(x, m * x + b, linewidth=2, c="black")
        g.ax_joint.set_xticks([])
        g.ax_joint.set_yticks([])
        g.ax_joint.set_xlabel(obj_sims.columns[0], fontsize=13)
        g.ax_joint.set_ylabel(obj_sims.columns[1], fontsize=13)
        g.ax_joint.annotate(
            "".join((r"$\rho$", " = ", str(rho))),
            (np.min(tril_1), np.max(tril_2)),
            fontsize=10,
        )
        g.savefig(os.path.join(PATH, "most_dissim_obj_pairs.jpg"))

    mods = [mod_1, mod_2]
    for i, most_dissim in enumerate(most_dissim_pairs):
        img_pairs = get_img_pairs(tril_inds, most_dissim)
        fig = plt.figure(figsize=(18, 12), dpi=100)
        ax = plt.subplot(111)

        for spine in ax.spines:
            ax.spines[spine].set_color(colors[i + 1])
            ax.spines[spine].set_linewidth(1.75)

        ax.imshow(img_pairs)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(mods[i])
        plt.savefig(
            os.path.join(
                PATH,
                "".join(
                    (
                        "most_dissim_obj_pairs",
                        "_",
                        "_".join(mods[i].lower().split()),
                        ".jpg",
                    )
                ),
            )
        )
        plt.close()


def plot_most_dissim_dims(
    plots_dir, human_dnn_corrs, behavior_images, weights_human, weights_dnn, top_k=4
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

            path_i = behavior_images[img_indices[0]]
            path_j = behavior_images[img_indices[1]]
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
            plt.close()


def find_rank_transformed_dissimilarities(
    weights_human, weights_dnn, behavior_images, plots_dir, topk=4
):
    """Find the most dissimilar pairs of images for each embedding based on rank transformed values."""
    n_objects = weights_human.shape[0]
    n_dim = min(weights_human.shape[1], weights_dnn.shape[1])

    for dim_idx in range(n_dim):

        dim_path = os.path.join(
            plots_dir, "human_dnn_comparison", str(dim_idx).zfill(2)
        )
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
                path_i = behavior_images[img_indices[i]]
                img_i = resize(io.imread(path_i), (400, 400))
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                axes[i].imshow(img_i)

            identifier = identifier.replace(" ", "_")
            fname = os.path.join(dim_path, f"{identifier}.png")
            fig.tight_layout()
            fig.savefig(fname, dpi=300, bbox_inches="tight")
            plt.close()


def plot_mind_machine_corrs(mind_machine_corrs, out_path, **kwargs):    
    plt.rc('font', family='sans-serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.rc('axes', labelsize=8)
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.set_context("paper")
    sns.set_style("whitegrid")
    sns.lineplot(x=range(len(mind_machine_corrs)), y=mind_machine_corrs, ax=ax)
    ax.lines[0].set_color("black")
    ax.lines[0].set_linewidth(1)
    
    ax.set_xlabel("Human Embedding Dimension")
    ax.set_ylabel("Highest Pearson's r \nwith DNN")
    fig.tight_layout()
    fig.savefig(os.path.join(out_path, "human_dnn_dimension_correlation.pdf"), 
                dpi=450, bbox_inches="tight", pad_inches=0.05)

def run_embedding_analysis(
    human_embedding, dnn_embedding, behavior_images, out_path="./plots"
):
    """Compare human and DNN performance on the same task."""

    # # Get rsm matrices
    # rsm_dnn = correlation_matrix(dnn_embedding)
    # rsm_human = correlation_matrix(human_embedding)

    # rho = correlate_rsms(rsm_dnn, rsm_human, "correlation")
    # print("Correlation between human and DNN embeddings: {}".format(rho))

    # Whether to perform mind-machine comparison with VGG 16 dimensions that allow for duplicate or unique latent dimensions
    human_sorted_indices, dnn_sorted_indices, mind_machine_corrs, all_dims = compare_modalities(human_embedding, dnn_embedding, duplicates=True, all_dims=True)


    plot_mind_machine_corrs(mind_machine_corrs, out_path, color="black", linewidth=2)


    # NOTE This does not make sense how I plot it right now
    # # Make a seaborn heatmap of the correlation matrix all_dims
    # fig, ax = plt.subplots()
    # n_human_dims = human_embedding.shape[1]
    # all_dims = all_dims[:, :n_human_dims]

    # # Generate a mask for the upper triangle
    # mask = np.triu(np.ones_like(all_dims, dtype=bool))

    # # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # sns.heatmap(all_dims, mask=mask, cmap=cmap, center=0,
    #         square=True, linewidths=.5, cbar_kws={"shrink": .5})
    # ax.set_xlabel("VGG 16")
    # ax.set_ylabel("Human")
    # fig.tight_layout()
    # fig.savefig(os.path.join(out_path, "human_dnn_dimension_corr_matrix.png"), dpi=300)


    # print("Plot density scatters for each object category")
    # plot_density_scatters(plots_dir=plot_dir_comparison, behavior_images=ref_images, rsm_1=rsm_human, rsm_2=rsm_dnn,
    #   mod_1='Human Behavior', mod_2='VGG 16', concepts=concepts, top_k=20)

    # print("Plot most dissimilar object pairs")
    # plot_most_dissim_pairs(plots_dir=out_path, behavior_images=behavior_images, rsm_1=rsm_human, rsm_2=rsm_dnn,
    #                        mod_1='Human Behavior', mod_2='VGG 16', top_k=20)

    human_embedding_sorted = human_embedding[:, human_sorted_indices]
    dnn_embedding_sorted = dnn_embedding[:, dnn_sorted_indices]

    # plot_most_dissim_dims(out_path, mind_machine_corrs, behavior_images, human_embedding_sorted, dnn_embedding_sorted)
    # find_rank_transformed_dissimilarities(human_embedding_sorted, dnn_embedding_sorted, behavior_images, out_path, topk=4)


if __name__ == "__main__":
    args = parser.parse_args()
    run_embedding_analysis(
        args.human_embedding_path, args.dnn_embedding_path, args.feature_path
    )
