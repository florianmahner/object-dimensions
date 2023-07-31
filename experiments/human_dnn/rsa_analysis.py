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
    correlation_matrix,
)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_context("paper", font_scale=2)
sns.set_style("white")

from tqdm import tqdm
from itertools import combinations
from typing import Tuple, Dict
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


def load_concepts(path: str = "./data/misc/category_mat_manual.tsv") -> pd.DataFrame:
    concepts = pd.read_csv(path, encoding="utf-8", sep="\t")
    return concepts


def plot_rsm(rsm: np.ndarray, fname: str) -> None:
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


def fast_rsm_correlations(human: np.ndarray, rsm_dnn: np.ndarray) -> float:
    # NOTE / TODO Check if we don't want to generally use the DNN ground truth
    rsm_human = rsm_pred_torch(human, return_type="torch")
    # rsm_dnn = rsm_pred_torch(dnn, return_type="torch")

    corr = correlate_rsms_torch(rsm_human, rsm_dnn)
    corr = corr.detach()
    return corr


def compute_histogram_of_fits(
    human: np.ndarray,
    dnn_rsm: np.ndarray,
    out_path: str,
    ncomb: int = 10_000,
    run_analysis: bool = True,
) -> None:
    """Sample combinations of `ncomb` dimensions for human and DNN embeddings that
    best correlate with each other"""
    ndims = human.shape[1]
    combs = list(combinations(range(ndims), 3))
    random.seed(0)
    random.shuffle(combs)
    combs = combs[:ncomb]
    ncombs = len(combs)
    corrs = torch.zeros(ncombs)
    fname = os.path.join(out_path, "correlations_combinations.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    human = torch.tensor(human, device=device)
    dnn_rsm = torch.tensor(dnn_rsm, device=device)

    # Iterate across all combinations and correlate the RSMs
    if run_analysis:
        pbar = tqdm(total=ncombs)
        for i, comb in enumerate(combs):
            pbar.set_description(f"Combination {i} / {ncombs}")
            pbar.update(1)
            human_comb = human[:, comb]
            corrs[i] = fast_rsm_correlations(
                human_comb, dnn_rsm
            )  # we always use the ground truth DNN RSM

        corrs = corrs.cpu().numpy()
        human_sub = human[:, :3]
        baseline_human = rsm_pred_torch(human_sub, return_type="torch")
        baseline_corr = correlate_rsms_torch(baseline_human, dnn_rsm)
        baseline_corr = baseline_corr.detach().cpu().numpy()

        with open(fname, "wb") as f:
            pickle.dump((corrs, baseline_corr), f)

    with open(fname, "rb") as f:
        corrs, baseline_corr = pickle.load(f)

    # Plot the histogram of correlations
    fig, ax = plt.subplots(1)
    sns.histplot(corrs, ax=ax, bins=100)
    sns.despine()
    ax.set_xlabel("Pearson r between RSMs")
    ax.set_ylabel("Count")

    # Make a horizontal line at the baseline correlation
    ax.axvline(baseline_corr, color="r", linestyle="--", label="Dimensions 1-3")
    ax.legend()
    fname = os.path.join(out_path, "histogram_correlations_combinations.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0.05)
    best_combination = combs[np.argmax(corrs)]
    best_corr = np.max(corrs)
    print(f"Best combination: {best_combination}, correlation: {best_corr}")


def calculate_rsm_corrs(
    weights: np.ndarray,
    weights_sum: np.ndarray,
    rsm: np.ndarray,
    dim_index: int,
    corr_type: str = "pearson",
    cumulative: bool = True,
) -> Tuple[float, float]:
    """Calculate the correlation between the RSMs for the given weights"""
    corr_gt = subset_rsm_analysis(
        weights, rsm, dim_index, corr=corr_type, cumulative=cumulative
    )
    corr_sum = subset_rsm_analysis(
        weights_sum, rsm, dim_index, corr=corr_type, cumulative=cumulative
    )
    return corr_gt, corr_sum


def plot_rsa_across_dims(rsa_dict: Dict[str, np.ndarray], out_path: str) -> None:
    """Plot the RSA across dimensions for the given weights"""
    df = pd.DataFrame(rsa_dict)
    # Rename the columns
    df = df.rename(
        columns={
            "dnn": "DNN",
            "human": "Human",
            "sum_dnn": "Sum DNN",
            "sum_human": "Sum Human",
            "individual_human": "Individual Human",
            "individual_dnn": "Individual DNN",
        }
    )
    df = df.reset_index()
    df = df.rename(columns={"index": "Cumulative Dimension"})
    df = df.melt(
        id_vars="Cumulative Dimension", var_name="Type", value_name="Highest Pearson r"
    )
    df_dnn = df[df["Type"].isin(["DNN", "Sum DNN"])]
    df_human = df[df["Type"].isin(["Human", "Sum Human"])]
    df_both = df[df["Type"].isin(["DNN", "Human"])]
    df_individual = df[df["Type"].isin(["Individual DNN", "Individual Human"])]

    kwargs = {
        "x": "Cumulative Dimension",
        "y": "Highest Pearson r",
        "hue": "Type",
        "style": "Type",
    }

    # Make all figures at once
    fnames = [
        "rsa_across_dims_dnn",
        "rsa_across_dims_human",
        "rsa_across_dims_both",
        "rsa_across_dims_individual",
    ]
    for data, fname in zip([df_dnn, df_human, df_both, df_individual], fnames):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        sns.lineplot(
            data=data,
            ax=ax,
            **kwargs,
        )
        sns.despine()
        for ext in ["png", "pdf"]:
            fig.savefig(
                os.path.join(
                    out_path,
                    fname + "." + ext,
                ),
                bbox_inches="tight",
                pad_inches=0.05,
            )


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

    # Plot the histogram of correlations
    compute_histogram_of_fits(
        weights_human,
        rsm_dnn_ground_truth,
        out_path,
        ncomb=1000,
        run_analysis=True,
    )

    breakpoint()

    weights_sum_human = correlate_modalities(
        human_embedding,
        dnn_embedding,
        base="human",
        duplicates=True,
        sort_by_corrs=False,
    )[0]

    weights_sum_dnn = correlate_modalities(
        human_embedding,
        dnn_embedding,
        base="dnn",
        duplicates=True,
        sort_by_corrs=False,
    )[0]

    rsm_human_reconstructed = rsm_pred_torch(weights_human)
    keys = [
        "dnn",
        "human",
        "sum_dnn",
        "sum_human",
        "individual_human",
        "individual_dnn",
    ]
    rsm_corrs = {key: np.zeros(ndims) for key in keys}

    if run_analysis:
        for i in range(ndims):
            print(f"Calculating RSM correlation for dim {i}", end="\r")

            rsm_corrs["dnn"][i], rsm_corrs["sum_dnn"][i] = calculate_rsm_corrs(
                weights_dnn,
                weights_sum_dnn,
                rsm_human_reconstructed,
                i,
            )

            rsm_corrs["human"][i], rsm_corrs["sum_human"][i] = calculate_rsm_corrs(
                weights_human,
                weights_sum_human,
                rsm_dnn_ground_truth,
                i,
            )

            rsm_corrs["individual_human"][i] = subset_rsm_analysis(
                weights_human,
                rsm_dnn_ground_truth,
                i,
                cumulative=False,
            )
            rsm_corrs["individual_dnn"][i] = subset_rsm_analysis(
                weights_dnn,
                rsm_human_reconstructed,
                i,
                cumulative=False,
            )

        with open(os.path.join(out_path, "rsa_across_dims.pkl"), "wb") as f:
            pickle.dump(rsm_corrs, f)

    with open(os.path.join(out_path, "rsa_across_dims.pkl"), "rb") as f:
        rsm_corrs = pickle.load(f)

    # Plot the RSA across dimensions
    plot_rsa_across_dims(rsm_corrs, out_path)

    # Baseline dimensions we take into account
    n_comparisons = 3

    # Calculate the RSMs
    rsm_top_human = rsm_pred_torch(weights_human[:, :n_comparisons])
    rsm_other_human = rsm_pred_torch(weights_human[:, n_comparisons:])
    rsm_all_human = rsm_pred_torch(weights_human)

    # Calculate the correlations
    corr_top = correlate_rsms(rsm_top_human, rsm_dnn_ground_truth, "pearson")
    corr_other = correlate_rsms(rsm_other_human, rsm_dnn_ground_truth, "pearson")
    corr_all = correlate_rsms(rsm_all_human, rsm_dnn_ground_truth, "pearson")

    print("Corr top {}: {}".format(n_comparisons, corr_top))
    print("Corr other {}: {}".format(ndims - n_comparisons, corr_other))
    print("Corr all {}: {}".format(ndims, corr_all))


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
    features = load_deepnet_activations(args.feature_path, center=True, relu=True)
    features = features[indices]

    rsm_truth_dnn = correlation_matrix(dnn_embedding)
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
        rsm_human,
        rsm_truth_dnn,
        concepts,
    )
    for rsm, name in zip([rsm_truth_dnn, rsm_human], ["dnn", "human"]):
        fname = os.path.join(plot_dir, f"{name}_rsm.jpg")
        plot_rsm(rsm, fname)
