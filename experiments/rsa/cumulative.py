""" This script finds how many DNN dimensions are required to explain 95% of the variance in the human RSM."""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from object_dimensions.utils import (
    pairiwise_correlate_dimensions,
    rsm_pred_torch,
    correlate_rsms,
)


def find_variance_threshold(cumulative_corrs, threshold=0.95):
    """Find the number of dimensions needed to explain the threshold variance"""
    cumulative_variance = np.cumsum(cumulative_corrs**2)
    dimension_idx = np.where(cumulative_variance >= threshold)[0][0]
    return dimension_idx


def run_cumulative_rsa(
    human_embedding,
    dnn_embedding,
):

    num_human_dims = human_embedding.shape[1]
    num_dnn_dims = dnn_embedding.shape[1]

    weights_dnn, weights_human = pairiwise_correlate_dimensions(
        dnn_embedding,
        human_embedding,
        duplicates=True,
        sort_by_corrs=True,
        return_corrs=False,
    )

    rsm_human = rsm_pred_torch(weights_human, return_type="numpy")
    cumulative_rsa_corrs = np.zeros(num_human_dims)

    for i in range(num_dnn_dims):
        w_i = weights_dnn[:, i][:, np.newaxis]
        rsm_dnn_i = rsm_pred_torch(w_i)
        corr_dnn_human = correlate_rsms(rsm_human, rsm_dnn_i, "pearson")
        cumulative_rsa_corrs[i] = corr_dnn_human

    return cumulative_rsa_corrs


def plot_cumulative_rsa(cumulative_rsa_corrs) -> None:
    """Plot the RSA across dimensions for the given weights"""

    color = sns.color_palette()[3]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    sns.lineplot(
        x=range(len(cumulative_rsa_corrs)),
        y=cumulative_rsa_corrs,
        errorbar=("sd", 2.5758),
        dashes=False,
        ax=ax,
        color=color,
        lw=1,
    )
    sns.despine(offset=10)

    dimension_explained_variance = find_variance_threshold(
        cumulative_rsa_corrs, threshold=0.95
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
    # Plot the 95% variance explained line
    ax.axvline(
        dimension_explained_variance, ls="--", color="black", lw=1, label=r"95% $R^2$"
    )
    ax.legend(loc="best", frameon=False)
    return fig
