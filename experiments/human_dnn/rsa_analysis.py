#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" This script computes all RSA analyses between human and DNN representations."""

import numpy as np
import torch
import pickle
import os
from experiments.human_dnn.reconstruction_accuracy import rsm_pred_torch
from experiments.human_dnn.compare_modalities import correlate_modalities
from scipy.stats import pearsonr, rankdata, spearmanr
from object_dimensions.utils.utils import (
    correlation_matrix,
    correlate_rsms,
    load_sparse_codes,
    load_image_data,
    create_path_from_params,
)
import matplotlib.pyplot as plt
from object_dimensions import ExperimentParser
import pandas as pd
import seaborn as sns


parser = ExperimentParser(
    description="Compare human and DNN performance on the same task."
)
parser.add_argument("--human_path", type=str, help="Path to human embedding matrix.")
parser.add_argument("--dnn_path", type=str, help="Path to DNN embedding matrix.")
parser.add_argument(
    "--img_root", type=str, help="Path to VGG feature matrix and filenames"
)
parser.add_argument(
    "--concept_path",
    type=str,
    default="./data/misc/category_mat_manual.tsv",
    help="Path to concept matrix",
)
parser.add_argument(
    "--corr_type",
    type=str,
    choices=["pearson", "spearman"],
    help="Type of correlation to use",
)
parser.add_argument("--run_analysis", action="store_true", help="Run analysis")


def load_concepts(path="./data/misc/category_mat_manual.tsv"):
    concepts = pd.read_csv(path, encoding="utf-8", sep="\t")
    return concepts


def plot_rsm(rsm, fname):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis("off")
    ax.imshow(rsm)
    fig.savefig(fname, pad_inches=0, bbox_inches="tight", dpi=450)
    plt.close(fig)


def filter_rsm_concepts(rsm_human, rsm_dnn, concepts, plot_dir):
    """Sort human and DNN by their assigned concept categories, so that
    objects belonging to the same concept are grouped together in the RSM"""

    def get_singletons(concepts):
        set_union = concepts.sum(axis=1)
        unique_memberships = np.where(set_union > 1.0, 0.0, set_union).astype(bool)
        singletons = concepts.iloc[unique_memberships, :]
        non_singletons = concepts.iloc[~unique_memberships, :]
        return singletons, non_singletons

    def sort_singletons(singletons):
        return np.hstack(
            [
                singletons[singletons.loc[:, concept] == 1.0].index
                for concept in singletons.keys()
            ]
        )

    singletons, non_singletons = get_singletons(concepts)
    singletons = sort_singletons(singletons)
    non_singletons = np.random.permutation(non_singletons.index)
    sorted_items = np.hstack((singletons, non_singletons))

    rsm_human = rsm_human[sorted_items, :]
    rsm_human = rsm_human[:, sorted_items]

    rsm_dnn = rsm_dnn[sorted_items, :]
    rsm_dnn = rsm_dnn[:, sorted_items]

    # NOTE Maybe we can still do something here?
    # triang = np.tril_indices(rsm_human.shape[0], k=-1)
    # # rank sort rsm
    # rsm_human[triang] = rankdata(rsm_human[triang])
    # rsm_dnn[triang] = rankdata(rsm_dnn[triang])

    # rsm_human = rsm_human + rsm_human.T
    # rsm_dnn = rsm_dnn + rsm_dnn.T

    # # rsm_human = np.fill_diagonal(rsm_human, 1.0)
    # # rsm_dnn = np.fill_diagonal(rsm_dnn, 1.0)

    rsm_human = rankdata(rsm_human).reshape(rsm_human.shape)
    rsm_dnn = rankdata(rsm_dnn).reshape(rsm_dnn.shape)

    return rsm_human, rsm_dnn


def corr_rsm_across_dims(human, dnn, dim_index, corr="pearson", cumulative=False):
    """Compute for a subset of dimensions the reconstructed similarity matrix.
    If cumulative, we select all embedding dimensions up to the index, otherwise
    we only select the embedding dimension at the index to construct and correlate
    both RSMs"""
    if cumulative:
        dim_h = human[:, : dim_index + 1]
        dim_d = dnn[:, : dim_index + 1]

    else:
        dim_h = human[:, dim_index].reshape(-1, 1)
        dim_d = dnn[:, dim_index].reshape(-1, 1)

    rsm_h = rsm_pred_torch(dim_h)
    rsm_d = rsm_pred_torch(dim_d)
    corr_hd = correlate_rsms(rsm_h, rsm_d, corr)
    return corr_hd


def savefig(fig, path):
    for ext in ["png", "pdf"]:
        fig.savefig(
            path + "." + ext,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )

    plt.close(fig)


def plot_rsm_across_dims(
    human_embedding,
    dnn_embedding,
    out_path,
    run_analysis=True,
):
    ndims = min(human_embedding.shape[1], dnn_embedding.shape[1])
    weights_human, weights_dnn, _ = correlate_modalities(
        human_embedding, dnn_embedding, duplicates=True, sort_by_corrs=True
    )

    weights_human_sorted_human, weights_dnn_sorted_human, _ = correlate_modalities(
        human_embedding,
        dnn_embedding,
        base="human",
        duplicates=True,
        sort_by_corrs=False,
    )

    weights_dnn_sorted_dnn, weights_human_sorted_dnn, _ = correlate_modalities(
        human_embedding, dnn_embedding, base="dnn", duplicates=True, sort_by_corrs=False
    )

    rsm_dims = np.zeros(ndims)
    rsm_dims_human_sorted = np.zeros(ndims)
    rsm_dims_dnn_sorted = np.zeros(ndims)
    rsm_dims_individual = np.zeros(ndims)

    if run_analysis:
        for i in range(ndims):
            print("Calculating RSM correlation for dim {}".format(i), end="\r")
            corr_dim = corr_rsm_across_dims(
                weights_human, weights_dnn, i, corr="pearson", cumulative=True
            )
            corr_dim_human = corr_rsm_across_dims(
                weights_human_sorted_human,
                weights_dnn_sorted_human,
                i,
                corr="pearson",
                cumulative=True,
            )
            corr_dim_dnn = corr_rsm_across_dims(
                weights_dnn_sorted_dnn,
                weights_human_sorted_dnn,
                i,
                corr="pearson",
                cumulative=True,
            )

            corr_individual = corr_rsm_across_dims(
                weights_human, weights_dnn, i, corr="pearson", cumulative=False
            )

            rsm_dims[i] = corr_dim
            rsm_dims_human_sorted[i] = corr_dim_human
            rsm_dims_dnn_sorted[i] = corr_dim_dnn
            rsm_dims_individual[i] = corr_individual

        dict = {
            "rsm_dims": rsm_dims,
            "rsm_dims_human_sorted": rsm_dims_human_sorted,
            "rsm_dims_dnn_sorted": rsm_dims_dnn_sorted,
            "rsm_dims_individual": rsm_dims_individual,
        }

        with open(out_path + "rsm_across_dims.pkl", "wb") as f:
            pickle.dump(dict, f)

    with open(out_path + "rsm_across_dims.pkl", "rb") as f:
        file = pickle.load(f)
        rsm_dims = file["rsm_dims"]
        rsm_dims_human_sorted = file["rsm_dims_human_sorted"]
        rsm_dims_dnn_sorted = file["rsm_dims_dnn_sorted"]
        rsm_dims_individual = file["rsm_dims_individual"]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    labels = ["Correlation", "Human Sum", "DNN Sum"]
    colors = ["r", "b", "g"]
    targets = [rsm_dims, rsm_dims_human_sorted, rsm_dims_dnn_sorted]

    for l, c, y in zip(labels, colors, targets):
        sns.lineplot(x=range(ndims), label=l, color=c, alpha=0.6, ax=ax)
    ax.set_xlabel("Number of included dimensions")
    ax.set_ylabel(r"Pearson r between RSMs")
    ax.set_ylim(0, 0.65)

    ax.set_xticks(range(0, ndims - 1, 6))
    ax.set_xticklabels(range(1, ndims, 6))
    ax.legend(title="Sorting method")
    sns.despine()

    savefig(fig, os.path.join(out_path, "rsm_corrs_across_dims"))

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    sns.lineplot(
        x=range(ndims),
        y=rsm_dims_individual,
        color="b",
        alpha=0.6,
        ax=ax,
    )
    ax.set_xlabel("Human Embedding Dimension")
    ax.set_ylabel(r"Pearson r between RSMs")
    ax.set_ylim(0, 0.8)
    ax.set_xticks(range(0, ndims - 1, 6))
    ax.set_xticklabels(range(1, ndims, 6))
    sns.despine()
    savefig(fig, os.path.join(out_path, "rsm_corrs_individual"))


if __name__ == "__main__":
    """Compare the human and DNN embeddings"""

    args = parser.parse_args()
    dnn_embedding, dnn_var = load_sparse_codes(args.dnn_path, with_var=True, relu=True)
    human_embedding, human_var = load_sparse_codes(
        args.human_path, with_var=True, relu=True
    )

    plot_dir = create_path_from_params(args.dnn_path, "analyses", "human_dnn", "rsa")
    print("Save all human dnn comparisons to {}".format(plot_dir))
    image_filenames, indices = load_image_data(args.img_root, filter_behavior=True)
    dnn_embedding = dnn_embedding[indices]
    rsm_dnn = rsm_pred_torch(dnn_embedding)
    rsm_human = rsm_pred_torch(human_embedding)

    corr = correlate_rsms(rsm_dnn, rsm_human, args.corr_type)
    print(
        "{} correlation between human and DNN embeddings: {}".format(
            args.corr_type.capitalize(), corr
        )
    )
    concepts = load_concepts(args.concept_path)

    rsm_human, rsm_dnn = filter_rsm_concepts(rsm_human, rsm_dnn, concepts, plot_dir)
    for rsm, name in zip([rsm_dnn, rsm_human], ["dnn", "human"]):
        fname = os.path.join(plot_dir, f"{name}_rsm.jpg")
        plot_rsm(rsm, fname)

    plot_rsm_across_dims(
        human_embedding,
        dnn_embedding,
        plot_dir,
        run_analysis=args.run_analysis,
    )
