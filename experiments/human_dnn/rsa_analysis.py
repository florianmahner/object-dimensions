#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" This script computes all RSA analyses between human and DNN representations."""

import numpy as np
import torch
import pickle
import random
import scipy.io as io
import os
from experiments.human_dnn.reconstruct_rsm import rsm_pred_torch
from experiments.rsm_reconstruction_analysis import rsm_pred_numba
from experiments.human_dnn.compare_modalities import correlate_modalities
from object_dimensions.utils import (
    correlate_rsms,
    correlate_rsms_torch,
    load_sparse_codes,
    load_image_data,
    load_deepnet_activations,
    create_path_from_params,
    correlation_matrix,
)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import rankdata
from scipy.stats import spearmanr


sns.set(font_scale=1.5)
sns.set_style("ticks")

from numba import njit, prange

from tqdm import tqdm
from typing import Tuple, Dict
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
    parser.add_argument(
        "--words48_path",
        type=str,
        default="./data/misc/words48.csv",
        help="Path to words48 file containing object categories used to fully sample things similarity matrix for noise ceilings",
    )
    parser.add_argument(
        "--human_rd_gt",
        type=str,
        help="Path to human RDM ground truth",
        default="./data/misc/rdm48_human.mat",
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


def plot_rsm_corrs(rsm_corrs, rsm_corrs_w_duplicates, out_path, **kwargs):
    # Create a DataFrame and reshape it to long format
    df = pd.DataFrame(
        {
            "Human Embedding RSM": range(len(rsm_corrs)),
            "Unique": rsm_corrs,
            "With Replacement": rsm_corrs_w_duplicates,
        }
    ).melt(
        "Human Embedding RSM",
        var_name="Pairing",
        value_name="Highest Pearson's r with DNN RSM",
    )
    # Create the figure
    fig, ax = plt.subplots(figsize=(6, 4))

    colors = sns.color_palette()

    colors = [colors[2], colors[3]]

    # Create a lineplot
    sns.lineplot(
        data=df,
        x="Human Embedding RSM",
        y="Highest Pearson's r with DNN RSM",
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
    ax.set_xlabel("Human Dimension RSM")
    ax.set_ylabel("Pearson's r with DNN\nDimension RSM")

    plt.legend(frameon=False)

    fig.tight_layout()

    # Save the figure
    for ext in [".png", ".pdf"]:
        fig.savefig(
            os.path.join(out_path, "human_dnn_individual_rsa{}".format(ext)),
            bbox_inches="tight",
            pad_inches=0.05,
            dpi=300,
            transparent=False,
        )

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
    base: np.ndarray,
    rsm_comp: np.ndarray,
    dim_index: int,
    corr: str = "pearson",
    cumulative: bool = False,
) -> float:
    """Compute for a subset of dimensions the reconstructed similarity matrix.
    If cumulative, we select all embedding dimensions up to the index, otherwise
    we only select the embedding dimension at the index to construct and correlate
    both RSMs"""
    if cumulative:
        dim_base = base[:, : dim_index + 1]
    else:
        dim_base = base[:, dim_index].reshape(-1, 1)

    rsm_base = rsm_pred_torch(dim_base)
    corr_hd = correlate_rsms(rsm_human, rsm_comp, corr)
    return corr_hd, rsm_base


def fast_rsm_correlations(human: np.ndarray, rsm_dnn: np.ndarray) -> float:
    # NOTE / TODO Check if we don't want to generally use the DNN ground truth
    rsm_human = rsm_pred_torch(human, return_type="torch")
    corr = correlate_rsms_torch(rsm_human, rsm_dnn)
    corr = corr.detach()
    return corr


def get_combination_subset(n_combs: int, n: int, r: int) -> np.ndarray:
    ctr = 0
    samples = set()
    while ctr < n_combs:
        sample = np.random.choice(n, size=r)
        sample = tuple(np.sort(sample))
        if sample not in samples:
            samples.add(tuple(sample))
            ctr += 1

    samples = np.array(list(samples))
    assert len(np.unique(samples, axis=0)) == n_combs
    return samples


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
    n_gt = 30  # cut off for ground truth

    combs = get_combination_subset(ncomb, ndims, n_gt)

    corrs = torch.zeros(ncomb)
    fname = os.path.join(out_path, "correlations_combinations.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    human = torch.tensor(human, device=device)
    dnn_rsm = torch.tensor(dnn_rsm, device=device)

    # Iterate across all combinations and correlate the RSMs
    if run_analysis:
        pbar = tqdm(total=ncomb)
        for i, comb in enumerate(combs):
            pbar.set_description(f"Combination {i} / {ncomb}")
            pbar.update(1)
            human_comb = human[:, comb]
            corrs[i] = fast_rsm_correlations(human_comb, dnn_rsm)

        corrs = corrs.cpu().numpy()
        human_sub = human[:, :n_gt]
        baseline_human = rsm_pred_torch(human_sub, return_type="torch")
        baseline_corr = correlate_rsms_torch(baseline_human, dnn_rsm)
        baseline_corr = baseline_corr.detach().cpu().numpy()

        with open(fname, "wb") as f:
            pickle.dump((corrs, baseline_corr), f)

    with open(fname, "rb") as f:
        corrs, baseline_corr = pickle.load(f)

    # Plot the histogram of correlations
    fig, ax = plt.subplots(1, figsize=(6, 4))
    sns.histplot(corrs, ax=ax, bins=50)
    sns.despine(offset=10)
    ax.set_xlabel("Pearson r between RSMs")
    ax.set_ylabel("Count")

    # Make a horizontal line at the baseline correlation
    ax.axvline(baseline_corr, color="r", linestyle="--", label="Dimensions 1-30")
    ax.legend()
    for fmt in ["png", "pdf"]:
        fname = os.path.join(out_path, f"histogram_correlations_combinations.{fmt}")
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
    corr_gt, rsm_corr = subset_rsm_analysis(
        weights, rsm, dim_index, corr=corr_type, cumulative=cumulative
    )
    corr_sum, rsm_sum = subset_rsm_analysis(
        weights_sum, rsm, dim_index, corr=corr_type, cumulative=cumulative
    )
    return corr_gt, corr_sum, rsm_corr, rsm_sum


def compute_bootstrap_corrs(
    rsms: Dict[str, np.ndarray], nboot: int = 100
) -> Dict[str, np.ndarray]:
    cumulative_dnn = rsms["dnn"]
    human_rsm_gt = rsms["human_reconstructed"]

    ndims = len(cumulative_dnn)
    n_objects = human_rsm_gt.shape[0]
    tril_inds = np.triu_indices(n_objects, k=1)
    human_rdv = human_rsm_gt[tril_inds]
    n = len(human_rdv)
    random.seed(0)
    inds = np.random.randint(0, n, size=(nboot, n))

    pbar = tqdm(total=ndims)
    boot_corrs = np.zeros((ndims, nboot))

    human_rdv_boot = human_rdv[inds]

    for i in range(ndims):
        pbar.update(1)

        dnn = cumulative_dnn[i]
        dnn_rdv = dnn[tril_inds]

        dnn_rdv_boot = dnn_rdv[inds]

        dnn_mean = np.mean(dnn_rdv_boot, axis=1, keepdims=True)
        human_mean = np.mean(human_rdv_boot, axis=1, keepdims=True)
        dnn_std = human_rdv_boot.std(axis=1)
        human_std = human_rdv_boot.std(axis=1)

        # Compute Pearson correlation coefficients
        boot_corrs[i, :] = (
            (human_rdv_boot - human_mean) * (dnn_rdv_boot - dnn_mean)
        ).sum(axis=1) / (n * human_std * dnn_std)

    boot_mean = boot_corrs.mean(axis=1)
    boot_std = boot_corrs.std(axis=1)

    return {"boot_corrs": boot_corrs, "boot_mean": boot_mean, "boot_std": boot_std}


def plot_rsa_across_dims(
    rsa_dict: Dict[str, np.ndarray], boot: Dict[str, np.ndarray], out_path: str
) -> None:
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

    # Replace tick label 0 with 1
    xticks = [10, 20, 30, 40, 50, 60]
    xticklabels = [str(x) for x in xticks]
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
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel("Pearson r to RSM")
        sns.despine(offset=10)
        for ext in ["png", "pdf"]:
            fig.savefig(
                os.path.join(
                    out_path,
                    fname + "." + ext,
                ),
                bbox_inches="tight",
                pad_inches=0.05,
            )

    # Plot the bootstrap for the human part only
    boot_df = pd.DataFrame(boot["boot_corrs"])
    boot_df = boot_df.T
    boot_df.reset_index(inplace=True)
    boot_df = boot_df.melt(
        id_vars="index", var_name="Cumulative Dimension", value_name="Pearson r"
    )

    color = sns.color_palette()[3]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    sns.lineplot(
        data=boot_df,
        x="Cumulative Dimension",
        y="Pearson r",
        errorbar=("sd", 2.5758),
        dashes=False,
        ax=ax,
        color=color,
        lw=1,
    )
    sns.despine(offset=10)
    # ax.set_xlabel("Cumulative Human Dimension")
    # ax.set_ylabel("Pearson's r to DNN RSM")
    ax.set_xlabel("Cumulative DNN Dimension")
    ax.set_ylabel("Pearson's r to Human RSM")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    max_pearson = boot["boot_mean"][-1]
    r2 = max_pearson**2
    pearson_90 = np.sqrt(r2 * 0.90)
    pearson_95 = np.sqrt(r2 * 0.95)
    pearson_99 = np.sqrt(r2 * 0.99)
    print("CI Bootstrap 90%: ", pearson_90)
    print("CI Bootstrap 95%: ", pearson_95)
    print("CI Bootstrap 99%: ", pearson_99)

    # Find dimension where the 90% CI is reached
    pearson_95_idx = np.where(boot["boot_mean"] >= pearson_95)[0][0]
    ax.axvline(pearson_95_idx, ls="--", color="black", lw=1, label=r"95% $R^2$")

    print("Number of dimensions need to explain 95% var", pearson_95_idx)
    ax.legend(loc="best", frameon=False)

    # fig.legend()

    fname = os.path.join(out_path, "rsa_across_dims_dnn_boot.{}")
    for ext in ["png", "pdf"]:
        fig.savefig(fname.format(ext), bbox_inches="tight", pad_inches=0.05)


def curve_fit_graph(rsa_corrs):
    """Experimental Test for Curve fitting the RSA corrs"""
    from scipy.optimize import curve_fit

    def inverted_exponential(x, a, b, c):
        return a * np.exp(-b * x) + c

    def derivative_inverted_exponential(x, a, b):
        return -a * b * np.exp(-b * x)

    y = rsa_corrs["human"]
    x = np.arange(1, len(y) + 1)
    popt, pcov = curve_fit(inverted_exponential, x, y)
    fit_y = inverted_exponential(x, *popt)

    fig, ax = plt.subplots(1)
    sns.lineplot(x=x, y=y, ax=ax, label="Human")
    sns.lineplot(x=x, y=fit_y, ax=ax, label="Fit")
    sns.despine()
    fig.savefig("test.png", dpi=300)

    fit_y_derivative = derivative_inverted_exponential(x, *popt[:-1])
    saturation_point = x[np.abs(fit_y_derivative) < 0.1]


def plot_rsm_across_dims(
    human_embedding,
    dnn_embedding,
    rsm_dnn,
    out_path,
    run_analysis=True,
):
    ndims = min(human_embedding.shape[1], dnn_embedding.shape[1])
    n_objects = human_embedding.shape[0]
    # weights_human, weights_dnn = correlate_modalities(
    #     human_embedding, dnn_embedding, duplicates=True, sort_by_corrs=True
    # )[:2]

    weights_dnn, weights_human = correlate_modalities(
        dnn_embedding, human_embedding, duplicates=True, sort_by_corrs=True
    )[:2]

    # Plot the histogram of correlations
    compute_histogram_of_fits(
        weights_human,
        rsm_dnn,
        out_path,
        ncomb=10_000,
        run_analysis=False,
    )

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

    ndims = 70
    rsm_corrs = {key: np.zeros(ndims) for key in keys}
    rsms = {key: np.zeros((ndims, n_objects, n_objects)) for key in keys}
    rsms["human_reconstructed"] = rsm_human_reconstructed

    run_analysis = False

    if run_analysis:
        for i in range(ndims):
            print(f"Calculating RSM correlation for dim {i}", end="\r")

            (
                rsm_corrs["dnn"][i],
                rsm_corrs["sum_dnn"][i],
                rsms["dnn"][i],
                rsms["sum_dnn"][i],
            ) = calculate_rsm_corrs(
                weights_dnn,
                weights_sum_dnn,
                rsm_human_reconstructed,
                i,
            )

            # (
            #     rsm_corrs["human"][i],
            #     rsm_corrs["sum_human"][i],
            #     rsms["human"][i],
            #     rsms["sum_human"][i],
            # ) = calculate_rsm_corrs(
            #     weights_human,
            #     weights_sum_human,
            #     rsm_dnn,
            #     i,
            # )

            # (
            #     rsm_corrs["individual_human"][i],
            #     rsms["individual_human"][i],
            # ) = subset_rsm_analysis(
            #     weights_human,
            #     rsm_dnn,
            #     i,
            #     cumulative=False,
            # )

            rsm_corrs["individual_dnn"][i], _ = subset_rsm_analysis(
                weights_dnn,
                rsm_human_reconstructed,
                i,
                cumulative=False,
            )
            rsms["individual_dnn"][i] = rsm_dnn

        with open(os.path.join(out_path, "rsa_across_dims.pkl"), "wb") as f:
            pickle.dump((rsms, rsm_corrs), f)

    with open(os.path.join(out_path, "rsa_across_dims.pkl"), "rb") as f:
        rsms, rsm_corrs = pickle.load(f)

    run_analysis = False
    if run_analysis:
        boot = compute_bootstrap_corrs(rsms, nboot=100)
        with open(os.path.join(out_path, "bootstrap_corrs.pkl"), "wb") as f:
            pickle.dump(boot, f)

    else:
        with open(os.path.join(out_path, "bootstrap_corrs.pkl"), "rb") as f:
            boot = pickle.load(f)

    # Plot the RSA across dimensions
    plot_rsa_across_dims(rsm_corrs, boot, out_path)

    # Baseline dimensions we take into account
    n_comparisons = [3, 5, 10]

    for ncomp in n_comparisons:
        # Calculate the RSMs
        rsm_top_human = rsm_pred_torch(weights_human[:, :ncomp])
        rsm_other_human = rsm_pred_torch(weights_human[:, ncomp:])
        rsm_all_human = rsm_pred_torch(weights_human)

        # Calculate the correlations
        corr_top = correlate_rsms(rsm_top_human, rsm_dnn, "pearson")
        corr_other = correlate_rsms(rsm_other_human, rsm_dnn, "pearson")
        corr_all = correlate_rsms(rsm_all_human, rsm_dnn, "pearson")

        print("Corr top {}: {}".format(ncomp, corr_top))
        print("Corr other {}: {}".format(ndims - ncomp, corr_other))
        print("Corr all {}: {}".format(ndims, corr_all))


@njit(parallel=False, fastmath=False)
def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    I, K = A.shape
    K, J = B.shape
    C = np.zeros((I, J))
    for i in prange(I):
        for j in prange(J):
            for k in prange(K):
                C[i, j] += A[i, k] * B[k, j]
    return C


@njit(parallel=False, fastmath=False)
def rsm_pred_numba(W: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """convert weight matrix corresponding to the mean of each dim distribution for an object into a RSM"""
    N = W.shape[0]
    S = matmul(W, W.T)
    S_e = np.exp(S)  # exponentiate all elements in the inner product matrix S
    rsm = np.zeros((N, N))
    for i in prange(N):
        for j in prange(i + 1, N):
            for k in indices:
                if k != i and k != j:
                    rsm[i, j] += S_e[i, j] / (S_e[i, j] + S_e[i, k] + S_e[j, k])

    # TODO I have hardcoded the 48 in here why??
    rsm /= 48
    rsm += rsm.T  # make similarity matrix symmetric
    np.fill_diagonal(rsm, 1)
    return rsm


def split_rsm(rsm):
    # Split the RSM into two halves based on the objects
    m = rsm.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)
    half_size = m // 2
    first_half_indices = indices[:half_size]
    second_half_indices = indices[half_size:]

    rsm_1 = np.corrcoef(rsm[first_half_indices, :][:, first_half_indices])
    rsm_2 = np.corrcoef(rsm[second_half_indices, :][:, second_half_indices])

    return rsm_1, rsm_2


def flatten_upper_triangle(matrix):
    # Get the upper triangle indices, excluding the diagonal
    upper_triangle_indices = np.triu_indices_from(matrix, k=1)
    return matrix[upper_triangle_indices]


def split_half_reliability(rsm: np.ndarray):
    rsm_1, rsm_2 = split_rsm(rsm)

    # Flatten the upper triangular parts of the correlation matrices
    rsm_1_flattened = flatten_upper_triangle(rsm_1)
    rsm_2_flattened = flatten_upper_triangle(rsm_2)

    # Compute the Pearson correlation
    split_half_corr, _ = pearsonr(rsm_1_flattened, rsm_2_flattened)

    # Spearman-Brown formula for reliability (i.e., noise ceiling)
    reliability = (2 * split_half_corr) / (1 + split_half_corr)
    return reliability


def run_rsa_on_individual_dimensions(weights_human, weights_dnn, sort_by_corrs=True):
    def compute_rsm(weights):
        num_dimensions = weights.shape[1]
        rsms = []
        for i in range(num_dimensions):
            print(f"Computing RSM for dimension {i}", end="\r")
            w = weights[:, i].reshape(-1, 1)
            rsm = rsm_pred_torch(w, return_type="numpy")
            rsms.append(rsm)
        return rsms

    weights = {"human": weights_human, "dnn": weights_dnn}
    weights_base = weights["human"]
    weights_comp = weights["dnn"]

    rsms_base = compute_rsm(weights_base)
    rsms_comp = compute_rsm(weights_comp)

    matching_corrs_with_duplicates = []
    matching_corrs_without_duplicates = []
    used_dims = set()

    for i, rsm_1 in enumerate(rsms_base):
        print(f"Comparing human dimension {i}", end="\r")

        rsm_corrs = np.array([correlate_rsms(rsm_1, rsm_2) for rsm_2 in rsms_comp])
        sorted_dim_corrs = np.argsort(-rsm_corrs)

        matching_corrs_with_duplicates.append(rsm_corrs[sorted_dim_corrs[0]])

        # Without duplicates
        for dim in sorted_dim_corrs:
            if dim not in used_dims:
                used_dims.add(dim)
                matching_corrs_without_duplicates.append(rsm_corrs[dim])
                break

        print(f"Best matching DNN dimension: {sorted_dim_corrs[0]}")

    if sort_by_corrs:
        matching_corrs_with_duplicates = np.array(matching_corrs_with_duplicates)
        matching_corrs_without_duplicates = np.array(matching_corrs_without_duplicates)

        sorted_indices_with = np.argsort(-matching_corrs_with_duplicates)
        sorted_indices_without = np.argsort(-matching_corrs_without_duplicates)

        matching_corrs_with_duplicates = matching_corrs_with_duplicates[
            sorted_indices_with
        ]
        matching_corrs_without_duplicates = matching_corrs_without_duplicates[
            sorted_indices_without
        ]

    return matching_corrs_with_duplicates, matching_corrs_without_duplicates


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

    # Do the rsa on individual dimensions
    rsa_corrs_individual_duplicates, rsa_corrs_individual = (
        run_rsa_on_individual_dimensions(
            human_embedding,
            dnn_embedding,
            sort_by_corrs=True,
        )
    )

    plot_rsm_corrs(rsa_corrs_individual, rsa_corrs_individual_duplicates, plot_dir)

    features = load_deepnet_activations(args.feature_path, center=True, relu=True)
    features = features[indices]

    rsm_truth_dnn = correlation_matrix(features)

    # rsm_truth_dnn = correlation_matrix(features)
    # from experiments.rsm_reconstruction_analysis import rsm_pred_numba

    # test = rsm_pred_numba(features)

    # rsm_recon_dnn = rsm_pred_numba(dnn_embedding)

    rsm_recon_dnn = rsm_pred_torch(dnn_embedding, return_type="numpy")

    rsm_human = rsm_pred_torch(human_embedding, return_type="numpy")

    corr_recon = correlate_rsms(rsm_recon_dnn, rsm_human, args.corr_type)
    corr_gt = correlate_rsms(rsm_truth_dnn, rsm_human, args.corr_type)

    corr_dnn_dnn_gt = correlate_rsms(rsm_truth_dnn, rsm_recon_dnn, args.corr_type)
    print("Pearson r between DNN and DNN ground truth: {}".format(corr_dnn_dnn_gt))

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
        rsm_recon_dnn,  # NOTE here is where i either use ground truth or reonstructed RSM.
        plot_dir,
        run_analysis=args.run_analysis,
    )

    # Note here is where we compute the RSMs for the human upper bound based on the similarity
    # of the concepts

    # NOTE -> I have taken the recon DNN embedding here.
    concepts = load_concepts(args.concept_path)
    rsm_human, rsm_truth_dnn = filter_rsm_by_concepts(
        rsm_human,
        rsm_recon_dnn,
        concepts,
    )

    words48 = pd.read_csv(args.words48_path, encoding="utf-8")

    cls_names = words48["Word"].values
    # sort aslphabveticaly
    sorted_indices = np.argsort(cls_names)
    cls_names = cls_names[sorted_indices]
    rdm_human_gt = io.loadmat(args.human_rd_gt)["RDM48_triplet"]
    rdm_human_gt = rdm_human_gt[np.ix_(sorted_indices, sorted_indices)]
    rsm_human_gt = 1 - rdm_human_gt

    # rsm_human_gt = rsm_human_gt[sorted_indices, :][:, sorted_indices]

    concept_path = "./data/misc/things_concepts.tsv"
    concepts = load_concepts(concept_path)

    # Concepts has a column of words. i want to find the indices of the words that are in the cls_names
    # and then sort the rsm by those indices.
    indices_48 = []

    def filter_concepts_cls_names(concepts, cls_names):
        """Filter by cls names"""
        cls_names = cls_names[(cls_names != "camera") & (cls_names != "file")]
        cls_names = np.append(cls_names, ["camera1", "file1"])
        # Replace white space with underscore in cls_names
        cls_names = [s.replace(" ", "_") for s in cls_names]
        regex_pattern = "|".join([rf"\b{s}\b" for s in cls_names])
        filtered_concepts = concepts[
            concepts["uniqueID"].str.contains(regex_pattern, regex=True)
        ]
        indices = filtered_concepts.index
        return indices

    indices_48 = filter_concepts_cls_names(concepts, cls_names)
    rsm_pred_human_filter = rsm_pred_numba(human_embedding, np.array(indices_48))
    rsm_pred_human_filter = rsm_pred_human_filter[np.ix_(indices_48, indices_48)]

    reliability = split_half_reliability(rsm_human_gt)
    print(f"Reliability of human RSM: {reliability}")

    true_rsm = rsm_human_gt.flatten()
    pred_rsm = rsm_pred_human_filter.flatten()

    from scipy.stats import spearmanr

    # NOTE this is where i calculate the explained variance but it does not work...
    true_pred_corr, _ = spearmanr(true_rsm, pred_rsm)

    corr_human_human_gt, pval = correlate_rsms(
        rsm_pred_human_filter, rsm_human_gt, args.corr_type, return_pval=True
    )

    # this below corrrelates the DNN filtered by the 48
    features48 = features[indices_48]
    rsm_truth_dnn = correlation_matrix(features48)
    dnn_embedding_48 = dnn_embedding[indices_48]
    recon_dnn_48 = rsm_pred_torch(dnn_embedding_48, return_type="numpy")

    # This below is the code to get the image to image similarity. This number has to be spearman
    # brown corrected

    print(
        f"Pearson r between human and human ground truth: {corr_human_human_gt}, pval {pval:.20f}"
    )

    for rsm, name in zip([rsm_truth_dnn, rsm_human], ["dnn", "human"]):
        fname = os.path.join(plot_dir, f"{name}_rsm.jpg")
        plot_rsm(rsm, fname)
