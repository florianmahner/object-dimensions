""" This script finds how many DNN dimensions are required to explain 95% of the variance in the human RSM."""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from objdim.utils import (
    rsm_pred_torch,
)
from tqdm import tqdm
from typing import Dict


def find_variance_threshold(cumulative_corrs, threshold=0.95):
    """Find the number of dimensions needed to explain the threshold variance"""
    cumulative_variance = cumulative_corrs**2
    largest_variance = cumulative_variance[-1]
    correlation_threshold = np.sqrt(largest_variance * threshold)
    dimension_idx = np.where(cumulative_corrs > correlation_threshold)[0][0]
    return dimension_idx


def bootstrap_rsm_correlation(
    rsm_human: np.ndarray, rsm_dnn: np.ndarray, nboot: int = 100
) -> Dict[str, np.ndarray]:
    """
    Compute bootstrap Pearson correlation coefficients between human and DNN Representational Similarity Matrices (RSMs).

    This function performs the following steps:
    1. Extracts the upper triangle (excluding the diagonal) from the input RSMs, converting them to 1D arrays.
    2. Generates bootstrap samples by resampling with replacement from the 1D arrays.
    3. Computes the mean and standard deviation for each bootstrap sample.
    4. Calculates the Pearson correlation coefficients for each pair of bootstrap samples from the human and DNN RSMs.

    Parameters:
    rsm_human (np.ndarray): A square matrix representing the human RSM.
    rsm_dnn (np.ndarray): A square matrix representing the DNN RSM.
    nboot (int, optional): The number of bootstrap samples to generate. Default is 100.

    Returns:
    Dict[str, np.ndarray]: A dictionary containing the bootstrap Pearson correlation coefficients.

    Notes:
    - The function assumes that the input RSMs are square matrices of the same size.
    - The random seed is set to 0 to ensure reproducibility of the bootstrap samples.

    Example:
    >>> rsm_human = np.array([[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]])
    >>> rsm_dnn = np.array([[1, 0.6, 0.4], [0.6, 1, 0.5], [0.4, 0.5, 1]])
    >>> result = compute_bootstrap_rsm_corrs(rsm_human, rsm_dnn, nboot=1000)
    >>> print(boot_corr.mean())
    """
    n_objects = rsm_human.shape[0]
    triu_inds = np.triu_indices(n_objects, k=1)
    human_rdv = rsm_human[triu_inds]
    dnn_rdv = rsm_dnn[triu_inds]

    n = len(human_rdv)
    inds = np.random.randint(0, n, size=(nboot, n))

    human_rdv_boot = human_rdv[inds]
    dnn_rdv_boot = dnn_rdv[inds]

    # Ensure that random sampling results in bootstrap samples with mean and std deviation
    human_mean = np.mean(human_rdv_boot, axis=1, keepdims=True)
    dnn_mean = np.mean(dnn_rdv_boot, axis=1, keepdims=True)
    human_std = np.std(human_rdv_boot, axis=1)
    dnn_std = np.std(dnn_rdv_boot, axis=1)

    # Compute Pearson correlation coefficients
    boot_corr = ((human_rdv_boot - human_mean) * (dnn_rdv_boot - dnn_mean)).sum(
        axis=1
    ) / (n * human_std * dnn_std)

    return boot_corr


def run_cumulative_rsa(
    human_embedding,
    dnn_embedding,
    nboot=50,
):
    """We run a bootstrapped version of the RSA calculation for each dimension of the DNN embedding. That is,
    we iteratively add dimensions from the DNN and then construct an RSM. We bootstrap entries from this
    RSM to get a sampling distribution of correlation coefficients between human and DNN RSM up to
    that dimension."""
    rsm_human = rsm_pred_torch(human_embedding)
    num_dnn_dims = dnn_embedding.shape[1]
    bootstrapped_cumulative_rsa_corrs = np.zeros((num_dnn_dims - 1, nboot))

    # we start at one to avoid the trivial case of a 1D embedding
    for i in range(1, num_dnn_dims):
        print("Computing RSA for dimension", i + 1)
        w_i = dnn_embedding[:, : i + 1]
        rsm_dnn_i = rsm_pred_torch(w_i)
        boot_corr = bootstrap_rsm_correlation(rsm_human, rsm_dnn_i, nboot=nboot)
        bootstrapped_cumulative_rsa_corrs[i - 1] = boot_corr

    return bootstrapped_cumulative_rsa_corrs


def plot_cumulative_rsa(bootstrapped_cumulative_rsa_corrs) -> None:
    """Plot the mean of the sampling distribution of bootstrapped cumulative RSM correlations
    alongside the standard deviation as error bars. Also plot a vertical line at the point where the
    DNN embedding explains 95% of the variance in the human RSM."""

    color = sns.color_palette()[3]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    # my boostrapped cumulative corrs are of shape ndims x nboot. make a line plot with error bars as the variance
    cumulative_mean = bootstrapped_cumulative_rsa_corrs.mean(axis=1)
    cumulative_std = bootstrapped_cumulative_rsa_corrs.std(axis=1)
    sns.lineplot(
        x=np.arange(1, len(cumulative_mean) + 1),
        y=cumulative_mean,
        color=color,
        ax=ax,
    )

    plt.fill_between(
        np.arange(1, len(cumulative_mean) + 1),
        cumulative_mean - cumulative_std,
        cumulative_std + cumulative_std,
        color=color,
        alpha=0.1,
    )

    sns.despine(offset=10)
    dimension_explained_variance = find_variance_threshold(
        cumulative_mean, threshold=0.95
    )
    print(
        "Number of dimensions needed to explain 95% variance",
        dimension_explained_variance,
    )

    xticks = [10, 20, 30, 40, 50, 60]
    xticklabels = [str(x) for x in xticks]
    ax.set(
        xlabel="Cumulative DNN Dimension",
        ylabel="Pearson's r to Human RSM",
        xticks=xticks,
        xticklabels=xticklabels,
    )
    ax.axvline(
        dimension_explained_variance, ls="--", color="black", lw=1, label=r"95% $R^2$"
    )
    ax.legend(loc="best", frameon=False)
    return fig
