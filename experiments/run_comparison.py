#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tqdm
import matplotlib

import numpy as np
import pandas as pd
import seaborn as sns
import skimage.io as io
import matplotlib.pyplot as plt

from skimage.transform import resize
from tomlparse import argparse
from scipy.stats import rankdata, pearsonr, spearmanr

from objdim.utils import (
    load_sparse_codes,
    load_image_data,
    correlate_rsms,
    create_results_path,
    rsm_pred_torch,
    pairwise_correlate_dimensions,
)


matplotlib.rcParams["font.sans-serif"] = "Arial"
matplotlib.rcParams["font.family"] = "sans-serif"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare human and DNN performance on the same task."
    )
    parser.add_argument(
        "--human_path", type=str, help="Path to human embedding matrix."
    )
    parser.add_argument("--dnn_path", type=str, help="Path to DNN embedding matrix.")
    parser.add_argument(
        "--img_root", type=str, help="Path to VGG feature matrix and filenames"
    )
    parser.add_argument(
        "--concept_path", type=str, help="Path to concept matrix and filenames"
    )
    parser.add_argument(
        "--diff_measure",
        type=str,
        choices=["rank", "signed"],
        default="signed",
        help="Measure to use for comparing the two modalities in the scatter plot",
    )
    return parser.parse_args()


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

        rho = spearmanr(tril_1, tril_2)[0].round(5)

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
        plt.close(fig)


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
        plt.close()

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
        plt.close(fig)


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
        path = os.path.join(plots_dir, "per_dim", dim_idx_str)
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
            plt.close(fig)

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
            plt.close(fig)


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

        for img_indices, identifier in zip(
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
            plt.close(fig)


def plot_mind_machine_corrs(
    mind_machine_corrs, mind_machine_corrs_w_duplicates, out_path, **kwargs
):
    # Create a DataFrame and reshape it to long format
    df = pd.DataFrame(
        {
            "Human Embedding Dimension": range(len(mind_machine_corrs)),
            "Unique": mind_machine_corrs,
            "With Replacement": mind_machine_corrs_w_duplicates,
        }
    ).melt(
        "Human Embedding Dimension",
        var_name="Pairing",
        value_name="Highest Pearson's r with DNN",
    )

    # Create the figure
    fig, ax = plt.subplots(figsize=(6, 4))

    colors = sns.color_palette()

    colors = [colors[2], colors[3]]

    # Create a lineplot
    sns.lineplot(
        data=df,
        x="Human Embedding Dimension",
        y="Highest Pearson's r with DNN",
        hue="Pairing",
        palette=colors,
        errorbar=("sd", 95),
        n_boot=1000,
        ax=ax,
    )

    sns.despine(offset=10)

    # Replace tick label 0 with 1
    xticks = [10, 20, 30, 40, 50, 60]
    xticklabels = [str(x) for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # Set the x- and y-labels
    ax.set_xlabel("Human Dimension")
    ax.set_ylabel("Pearson's r with DNN")

    plt.legend(frameon=False)

    # Save the figure
    for ext in [".png", ".pdf"]:
        fig.savefig(
            os.path.join(out_path, "human_dnn_dimension_correlation{}".format(ext)),
            bbox_inches="tight",
            dpi=300,
        )

    plt.close(fig)


def concat_images(image_list, topk=8, factor=2):
    images = []
    per_row = topk // factor
    for image in image_list:
        image = resize(io.imread(image), (400, 400))
        images.append(image)

    image_row1 = np.concatenate(images[:per_row], axis=1)
    if factor == 1:
        return image_row1
    elif factor == 2:
        image_row2 = np.concatenate(images[per_row : per_row * 2], axis=1)
        large_image = np.concatenate([image_row1, image_row2], axis=0)
        return large_image


def visualize_dims_across_modalities(
    plots_dir,
    ref_images,
    w_mod1,
    w_mod2,
    latent_dim=0,
    r=None,
    difference="absolute",
    top_k=8,
):
    # find both top k objects per modality and their intersection
    topk_mod1 = np.argsort(-w_mod1)[:top_k]
    topk_mod2 = np.argsort(-w_mod2)[:top_k]
    top_k_common = np.intersect1d(topk_mod1, topk_mod2)
    # find top k objects (i.e., objects with hightest loadings) in current latent dimension for each modality
    topk_imgs_mod1 = ref_images[topk_mod1]
    topk_imgs_mod2 = ref_images[topk_mod2]

    # calculate rank or absolute differences of weight coefficients between the (two) modalities
    if difference == "rank":
        rank_diff_mod1 = rankdata(-w_mod1) - rankdata(-w_mod2)
        rank_diff_mod2 = rankdata(-w_mod2) - rankdata(-w_mod1)
        most_dissim_imgs_mod1 = ref_images[np.argsort(rank_diff_mod1)[:top_k]]
        most_dissim_imgs_mod2 = ref_images[np.argsort(rank_diff_mod2)[:top_k]]
    elif difference == "signed":
        diff_mod1 = w_mod1 - w_mod2
        diff_mod2 = w_mod2 - w_mod1
        most_dissim_imgs_mod1 = ref_images[np.argsort(diff_mod1)[::-1][:top_k]]
        most_dissim_imgs_mod2 = ref_images[np.argsort(diff_mod2)[::-1][:top_k]]
    else:
        raise ValueError("Difference must be either 'rank' or 'signed'.")

    # Plot the images in a grid themselves
    def plot_image_grid(images):
        images = images[:6]
        fig, axs = plt.subplots(2, 3, figsize=(4, 6))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)
        for i, image in enumerate(images):
            ax = axs[i // 3, i % 3]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
            img = io.imread(image)
            ax.imshow(img)

        return fig

    path = os.path.join(plots_dir, "compare_modalities", difference + "_diff")
    if not os.path.exists(path):
        os.makedirs(path)

    for imgs, fnames in zip(
        [topk_imgs_mod1, topk_imgs_mod2, most_dissim_imgs_mod1, most_dissim_imgs_mod2],
        [
            f"latent_dim_{latent_dim}_topk_human",
            f"latent_dim_{latent_dim}_topk_dnn",
            f"latent_dim_{latent_dim}_most_dissim_human",
            f"latent_dim_{latent_dim}_most_dissim_dnn",
        ],
    ):
        fig = plot_image_grid(imgs)
        fig.savefig(
            os.path.join(
                plots_dir, "compare_modalities", difference + "_diff", f"{fnames}.pdf"
            ),
            bbox_inches="tight",
            pad_inches=0.0,
        )

        plt.close(fig)

    data = pd.DataFrame({"Human": w_mod1, "DNN": w_mod2})
    # Make empty list of colors
    data["colors"] = 0
    high_dnn_low_human = np.argsort(diff_mod1)[::-1][:top_k]
    high_human_low_dnn = np.argsort(diff_mod2)[::-1][:top_k]
    top_k_common = np.intersect1d(topk_mod1, topk_mod2)

    data.loc[topk_mod1, "colors"] = 1
    data.loc[topk_mod2, "colors"] = 2
    data.loc[high_dnn_low_human, "colors"] = 3
    data.loc[high_human_low_dnn, "colors"] = 4
    if len(top_k_common) > 0:
        data.loc[top_k_common, "colors"] = 5

    # Make an alphas array where all grey colors are 0.2 and all others are 1
    alphas = np.where(data["colors"] == 0, 0.2, 0.8)
    data["alphas"] = alphas

    fig, ax = plt.subplots(1, figsize=(6, 4))
    palette = sns.color_palette()

    blue, orange, green, red = palette[:4]
    magenta = palette[6]

    palette = {
        0: "grey",
        1: blue,
        2: red,
        3: orange,
        4: green,
        5: magenta,
    }

    corrs = pearsonr(w_mod1, w_mod2)[0]
    corr_str = r"$r$" + " = " + str(corrs.round(2))

    scatter = ax.scatter(
        data["Human"],
        data["DNN"],
        c=data["colors"].map(palette),  # Map your colors to the specified palette
        alpha=data["alphas"],  # Use the alphas column for individual alpha values
        s=100,
    )

    sns.despine(offset=10, ax=ax)

    ax.set_xlabel("Human Behavior")
    ax.set_ylabel("DNN")

    loc_x = np.min(w_mod1)
    loc_y = np.max(w_mod2)
    ax.annotate(corr_str, (loc_x, loc_y))
    ax.legend([], [], frameon=False)

    fig.savefig(
        os.path.join(
            path,
            "latent_dim_{}_scatter.pdf".format(latent_dim),
        ),
        bbox_inches="tight",
        pad_inches=0.05,
    )

    plt.close(fig)

    most_dissim_imgs_mod1 = concat_images(most_dissim_imgs_mod1, topk=top_k, factor=2)
    most_dissim_imgs_mod2 = concat_images(most_dissim_imgs_mod2, topk=top_k, factor=2)
    topk_imgs_mod1 = concat_images(topk_imgs_mod1, topk=top_k, factor=2)
    topk_imgs_mod2 = concat_images(topk_imgs_mod2, topk=top_k, factor=2)

    titles = [r"Human Behavior", r"VGG 16"]

    n_rows = 2
    n_cols = 3

    # set variables and initialise figure object
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4), dpi=300)
    y_labs = [r"Top $k$", r"Most dissimilar"]

    axes[0, 0].imshow(topk_imgs_mod1)
    axes[0, 0].set_title(titles[0])
    axes[0, 0].set_ylabel(y_labs[0])

    axes[0, 1].imshow(topk_imgs_mod2)
    axes[0, 1].set_title(titles[1])

    axes[1, 0].imshow(most_dissim_imgs_mod1)
    axes[1, 0].set_ylabel(y_labs[1])
    axes[1, 1].imshow(most_dissim_imgs_mod2)

    # Color differently based on the topk row or the most dissimlar row
    for i in range(n_rows):
        colors = np.array(["grey" for _ in range(len(w_mod1))])
        alphas = np.array([0.2 for _ in range(len(w_mod1))])
        if i == 0:
            colors[topk_mod1] = "r"
            colors[topk_mod2] = "b"
            alphas[topk_mod1] = 0.6
            alphas[topk_mod2] = 0.6
            if len(top_k_common) > 0:
                colors[top_k_common] = "m"
        else:
            if difference == "rank":
                colors[np.argsort(rank_diff_mod1)[:top_k]] = "r"
                colors[np.argsort(rank_diff_mod2)[:top_k]] = "b"
            else:
                colors[np.argsort(diff_mod1)[::-1][:top_k]] = "r"
                colors[np.argsort(diff_mod2)[::-1][:top_k]] = "b"
                alphas[np.argsort(diff_mod1)[::-1][:top_k]] = 0.7
                alphas[np.argsort(diff_mod2)[::-1][:top_k]] = 0.7

        axes[i, 2].scatter(w_mod1, w_mod2, c=colors, s=45, alpha=alphas)

    for ax in [axes[0, 2], axes[1, 2]]:
        ax.set_xlabel(r"Human Behavior")
        ax.set_ylabel(r"VGG 16")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    for ax in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ["left", "right", "top", "bottom"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle(r"Pearson r = {:.2f}".format(r))

    plt.savefig(
        os.path.join(
            path,
            "latent_dim_{}_across_modalities.pdf".format(latent_dim),
        ),
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)


def corr_rsm_across_dims(human, dnn, index, corr="pearson", cumulative=False):
    if cumulative:
        dim_h = human[:, :index]
        dim_d = dnn[:, :index]

    else:
        dim_h = human[:, index]
        dim_d = dnn[:, index]

    rsm_h = rsm_pred_torch(dim_h)
    rsm_d = rsm_pred_torch(dim_d)
    corr_hd = correlate_rsms(rsm_h, rsm_d, corr)

    return corr_hd


def run_embedding_analysis(
    human_path,
    dnn_path,
    img_root,
    concept_path="",
    difference="absolute",
):
    """Compare human and DNN performance on the same task."""
    dnn_embedding, dnn_var = load_sparse_codes(dnn_path, with_var=True, relu=True)
    human_embedding = load_sparse_codes(human_path, with_var=False, relu=True)

    out_path = create_results_path(dnn_path, "human_dnn_comparison")

    print("Save all human dnn comparisons to {}".format(out_path))
    image_filenames, indices = load_image_data(img_root, filter_behavior=True)
    dnn_embedding = dnn_embedding[indices]
    dnn_var = dnn_var[indices]

    human_embedding_sorted, dnn_embedding_sorted, corrs, matching_dims = (
        pairwise_correlate_dimensions(
            human_embedding, dnn_embedding, duplicates=False, sort_by_corrs=True
        )
    )

    corrs_w_duplicates, match_duplicates = pairwise_correlate_dimensions(
        human_embedding, dnn_embedding, duplicates=True, sort_by_corrs=True
    )[2:4]

    df = pd.DataFrame(
        {
            "Human Dimension": list(range(len(corrs))),
            "match_unique": matching_dims,
            "match_duplicates": match_duplicates,
        }
    )

    with open(os.path.join(out_path, "matching_dims.csv"), "w") as f:
        df.to_csv(f, index=False)

    plot_mind_machine_corrs(
        corrs,
        corrs_w_duplicates,
        out_path,
        color="black",
        linewidth=2,
    )

    for j, (w_b, w_dnn) in enumerate(
        zip(human_embedding_sorted.T[:50], dnn_embedding_sorted.T[:50])
    ):
        print("Vis dim across modalities for dim {}".format(j), end="\r")
        pearson = corrs[j]

        w_b /= np.max(w_b)
        w_dnn /= np.max(w_dnn)
        visualize_dims_across_modalities(
            out_path,
            image_filenames,
            w_b,
            w_dnn,
            latent_dim=j,
            r=pearson,
            difference=difference,
        )

    dnn_embedding, dnn_var = load_sparse_codes(dnn_path, with_var=True, relu=True)
    human_embedding = load_sparse_codes(human_path, with_var=False, relu=True)
    rsm_human = rsm_pred_torch(human_embedding)
    rsm_dnn = rsm_pred_torch(dnn_embedding)
    concepts = pd.read_csv(concept_path, encoding="utf-8", sep="\t")

    print("Plot density scatters for each object category")

    plot_density_scatters(
        plots_dir=out_path,
        behavior_images=image_filenames,
        rsm_1=rsm_human,
        rsm_2=rsm_dnn,
        mod_1="Human Behavior",
        mod_2="VGG 16",
        concepts=concepts,
        top_k=20,
    )

    print("Plot most dissimilar object pairs")
    plot_most_dissim_pairs(
        plots_dir=out_path,
        rsm_1=rsm_human,
        rsm_2=rsm_dnn,
        mod_1="Human Behavior",
        mod_2="VGG 16",
        top_k=20,
    )
    find_rank_transformed_dissimilarities(
        human_embedding_sorted, dnn_embedding_sorted, image_filenames, out_path, topk=4
    )


if __name__ == "__main__":
    args = parse_args()
    run_embedding_analysis(
        args.human_path,
        args.dnn_path,
        args.img_root,
        args.concept_path,
        args.diff_measure,
    )
