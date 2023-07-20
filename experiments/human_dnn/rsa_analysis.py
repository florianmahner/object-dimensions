#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" This script computes all RSA analyses between human and DNN representations."""

import numpy as np
import torch
import pickle
import random
import os
from experiments.human_dnn.reconstruction_accuracy import rsm_pred_torch
from experiments.human_dnn.compare_modalities import correlate_modalities
from object_dimensions.utils.utils import (
    correlate_rsms,
    correlate_rsms_torch,
    load_sparse_codes,
    load_image_data,
    load_deepnet_activations,
    create_path_from_params,
    save_figure,
)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_context("paper", font_scale=2)
sns.set_style("white")

from tqdm import tqdm
from itertools import combinations
from typing import Tuple
from scipy.stats import rankdata
from tomlparse import argparse


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
    parser.add_argument(
        "--feature_path",
        type=str,
        default="./features",
        help="Path to DNN features directory",
    )
    return parser.parse_args()


def load_concepts(path="./data/misc/category_mat_manual.tsv"):
    concepts = pd.read_csv(path, encoding="utf-8", sep="\t")
    return concepts


def plot_rsm(rsm, fname):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis("off")
    ax.imshow(rsm, cmap="viridis", interpolation="nearest")
    fig.savefig(fname, pad_inches=0, bbox_inches="tight", dpi=450)
    plt.close(fig)


def filter_rsm_by_concepts(
    rsm_human: np.ndarray, rsm_dnn: np.ndarray, concepts: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
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
    rsm_human = rankdata(rsm_human).reshape(rsm_human.shape)
    rsm_dnn = rankdata(rsm_dnn).reshape(rsm_dnn.shape)
    return rsm_human, rsm_dnn


def subset_rsm_analysis(
    human: np.ndarray,
    rsm_dnn: np.ndarray,
    dim_index: int,
    corr: str = "pearson",
    cumulative: bool = False,
) -> float:
    """Compute for a subset of dimensions the reconstructed similarity matrix.
    If cumulative, we select all embedding dimensions up to the index, otherwise
    we only select the embedding dimension at the index to construct and correlate
    both RSMs"""
    if cumulative:
        dim_human = human[:, : dim_index + 1]
    else:
        dim_human = human[:, dim_index].reshape(-1, 1)

    rsm_human = rsm_pred_torch(dim_human)
    corr_hd = correlate_rsms(rsm_human, rsm_dnn, corr)
    return corr_hd


def find_rsm_correlations(human: np.ndarray, dnn: np.ndarray, **kwargs):
    # NOTE / TODO Check if we don't want to generally use the DNN ground truth
    rsm_h = rsm_pred_torch(human, **kwargs)
    rsm_d = rsm_pred_torch(dnn, **kwargs)
    corr = correlate_rsms_torch(rsm_h, rsm_d)
    corr = corr.detach()
    return corr


def find_best_fit(
    human: np.ndarray,
    dnn: np.ndarray,
    out_path: str,
    run_analysis: bool = True,
    ncomb: int = 10_000,
) -> None:
    """Sample combinations of `ncomb` dimensions for human and DNN embeddings that
    best correlate with each other"""
    ndims = min(human.shape[1], dnn.shape[1])
    combs = list(combinations(range(ndims), 3))
    random.shuffle(combs)
    combs = combs[:ncomb]
    ncombs = len(combs)
    corrs = torch.zeros(ncombs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    human = torch.tensor(human, device=device)
    dnn = torch.tensor(dnn, device=device)

    pbar = tqdm(total=ncombs)
    # Iterate across all combinations and correlate the RSMs
    for i, comb in enumerate(combs):
        pbar.set_description(f"Combination {i} / {ncombs}")
        pbar.update(1)
        human_comb = human[:, comb]
        dnn_comb = dnn[:, comb]
        corrs[i] = find_rsm_correlations(human_comb, dnn_comb)

    corrs = corrs.cpu().numpy()

    # Compute the baselone correlation between the RSMs
    baseline_human = human[:, :3] @ human[:, :3].T
    baseline_dnn = dnn[:, :3] @ dnn[:, :3].T
    baseline_corr = correlate_rsms_torch(baseline_human, baseline_dnn, "pearson")
    baseline_corr = baseline_corr.cpu().numpy()

    # Plot the histogram of correlations
    fig, ax = plt.subplots(1)
    ax.hist(corrs, bins=100)
    ax.set_xlabel("Pearson r between RSMs")
    ax.set_ylabel("Count")

    # Make a horizontal line at the baseline correlation
    ax.axvline(baseline_corr, color="r", linestyle="--", label="Baseline")
    fname = os.path.join(out_path, "histogram_correlations_combinations.png")
    fig.savefig(fname, dpi=300)
    best_combination = combs[np.argmax(corrs)]
    best_corr = np.max(corrs)
    print(f"Best combination: {best_combination}, correlation: {best_corr}")


def plot_rsm_across_dims(
    human_embedding,
    dnn_embedding,
    rsm_dnn_ground_truth,
    out_path,
    run_analysis=True,
):
    ndims = min(human_embedding.shape[1], dnn_embedding.shape[1])
    weights_human, weights_dnn, _ = correlate_modalities(
        human_embedding, dnn_embedding, duplicates=True, sort_by_corrs=True
    )

    find_best_fit(weights_human, weights_dnn, out_path)

    weights_human_sorted_human, weights_dnn_sorted_human, _ = correlate_modalities(
        human_embedding,
        dnn_embedding,
        base="human",
        duplicates=True,
        sort_by_corrs=False,
    )
    rsm_corrs = np.zeros(ndims)
    rsm_corrs_human_sorted = np.zeros(ndims)
    rsm_corrs_individual = np.zeros(ndims)

    if run_analysis:
        for i in range(ndims):
            print("Calculating RSM correlation for dim {}".format(i), end="\r")
            corr_dim = subset_rsm_analysis(
                weights_human, rsm_dnn_ground_truth, i, corr="pearson", cumulative=True
            )

            corr_dim = subset_rsm_analysis(
                weights_human,
                rsm_dnn_ground_truth,
                i,
                cumulative=True,
            )

            corr_dim_human = subset_rsm_analysis(
                weights_human_sorted_human,
                rsm_dnn_ground_truth,
                i,
                cumulative=True,
            )

            corr_individual = subset_rsm_analysis(
                weights_human, rsm_dnn_ground_truth, i, cumulative=False
            )

            rsm_corrs[i] = corr_dim
            rsm_corrs_human_sorted[i] = corr_dim_human
            rsm_corrs_individual[i] = corr_individual

        dict = {
            "rsm_corrs": rsm_corrs,
            "rsm_corrs_human_sorted": rsm_corrs_human_sorted,
            "rsm_corrs_individual": rsm_corrs_individual,
        }

        with open(os.path.join(out_path, "rsa_across_dims.pkl"), "wb") as f:
            pickle.dump(dict, f)

    with open(os.path.join(out_path, "rsa_across_dims.pkl"), "rb") as f:
        file = pickle.load(f)
        rsm_corrs = file["rsm_corrs"]
        rsm_corrs_human_sorted = file["rsm_corrs_human_sorted"]
        rsm_corrs_individual = file["rsm_corrs_individual"]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    labels = ["Correlation", "Human Sum"]
    colors = ["r", "b", "g"]
    targets = [rsm_corrs, rsm_corrs_human_sorted]

    for l, c, y in zip(labels, colors, targets):
        sns.lineplot(x=range(ndims), y=y, label=l, color=c, alpha=0.6, ax=ax)
    ax.set_xlabel("Cumulative Human Dimension")
    ax.set_ylabel(r"Pearson r between RSMs")
    ax.set_ylim(0, 0.8)

    ax.set_xticks(range(0, ndims - 1, 6))
    ax.set_xticklabels(range(1, ndims, 6))
    ax.legend(title="Sorting method")
    ax.set_title("Cumulative Dimension Correlations")
    sns.despine()

    save_figure(fig, os.path.join(out_path, "rsm_corrs_across_dims"))

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    sns.lineplot(
        x=range(ndims),
        y=rsm_corrs_individual,
        color="b",
        alpha=0.6,
        ax=ax,
    )
    ax.set_xlabel("Individual Human Dimension")
    ax.set_ylabel(r"Pearson r between RSMs")
    ax.set_ylim(0, 0.8)
    ax.set_xticks(range(0, ndims - 1, 6))
    ax.set_xticklabels(range(1, ndims, 6))
    sns.despine()
    ax.set_title("Individual Dimension Correlations")
    save_figure(fig, os.path.join(out_path, "rsm_corrs_individual"))

    # Baseline dimensions we take into account
    n_comparisons = 3
    rsm_top_human = rsm_pred_torch(human_embedding[:, :n_comparisons])
    rsm_top_dnn = rsm_pred_torch(dnn_embedding[:, :n_comparisons])

    # NOTE look and check into corrs here tomorrow!
    # I feel like the plot shows that the first 25 are important, before it saturates!

    rsm_other_dnn = rsm_pred_torch(dnn_embedding[:, n_comparisons:])
    rsm_other_human = rsm_pred_torch(human_embedding[:, n_comparisons:])

    rsm_all_human = rsm_pred_torch(human_embedding)
    rsm_all_dnn = rsm_pred_torch(dnn_embedding)

    corr_top = correlate_rsms(rsm_top_human, rsm_dnn_ground_truth, "pearson")
    corr_other = correlate_rsms(rsm_other_human, rsm_other_dnn, "pearson")
    corr_all = correlate_rsms(rsm_all_human, rsm_dnn_ground_truth, "pearson")

    breakpoint()

    print("Corr top: {}".format(corr_top))
    print("Corr other: {}".format(corr_other))
    print("Corr all: {}".format(corr_all))


if __name__ == "__main__":
    """Compare the human and DNN embeddings"""
    args = parse_args()
    dnn_embedding, dnn_var = load_sparse_codes(args.dnn_path, with_var=True, relu=True)
    human_embedding, human_var = load_sparse_codes(
        args.human_path, with_var=True, relu=True
    )

    plot_dir = create_path_from_params(args.dnn_path, "analyses", "human_dnn", "rsa")
    print("Save all human dnn comparisons to {}".format(plot_dir))
    image_filenames, indices = load_image_data(args.img_root, filter_behavior=True)
    dnn_embedding = dnn_embedding[indices]
    features = load_deepnet_activations(args.feature_path, center=True)
    features = features[indices]

    rsm_truth_dnn = features @ features.T
    rsm_recon_dnn = rsm_pred_torch(dnn_embedding, return_type="numpy")
    rsm_human = rsm_pred_torch(human_embedding, return_type="numpy")

    corr_recon = correlate_rsms(rsm_recon_dnn, rsm_human, args.corr_type)
    corr_gt = correlate_rsms(rsm_truth_dnn, rsm_human, args.corr_type)

    for corr, identifier in zip(
        [corr_recon, corr_gt], ["Reconstructed", "Ground Truth"]
    ):
        print(
            "{} correlation between human and DNN {}: {}".format(
                args.corr_type.capitalize(), identifier, corr
            )
        )

    plot_rsm_across_dims(
        human_embedding,
        dnn_embedding,
        rsm_truth_dnn,
        plot_dir,
        run_analysis=args.run_analysis,
    )

    # NOTE -> I have taken the ground truth DNN embedding here.
    concepts = load_concepts(args.concept_path)
    rsm_human, rsm_truth_dnn = filter_rsm_by_concepts(
        rsm_human, rsm_truth_dnn, concepts, plot_dir
    )
    for rsm, name in zip([rsm_truth_dnn, rsm_human], ["dnn", "human"]):
        fname = os.path.join(plot_dir, f"{name}_rsm.jpg")
        plot_rsm(rsm, fname)
