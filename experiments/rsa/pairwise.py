import pandas as pd
import numpy as np
from objdim.utils import (
    correlate_rsms,
    rsm_pred_torch,
)
import seaborn as sns
import matplotlib.pyplot as plt


def compute_rsm_per_dimension(embedding):
    num_dimensions = embedding.shape[1]
    rsms = []
    print(f"Computing RSMs for {num_dimensions} dimensions")
    for i in range(num_dimensions):
        w = embedding[:, i].reshape(-1, 1)
        rsm = rsm_pred_torch(w)
        rsms.append(rsm)
    return rsms


def sort_corrs(corrs):
    sorted_indices = np.argsort(-corrs)
    return corrs[sorted_indices]


def pairwise_rsm_comparison(weights_human, weights_dnn, sort_by_corrs=True):
    """We do a pairwise comparison of the human and DNN embeddings. In principle, we walk through each
    dimension of the human embedding, build an RSM from it and then compare it to all other RSMs from
    all other dimensions of the DNN. We then take the best matching DNN dimension and store the correlation
    value. We do this for all dimensions and then return the correlation values
    """
    rsms_per_dim_human = compute_rsm_per_dimension(weights_human)
    rsms_per_dim_dnn = compute_rsm_per_dimension(weights_dnn)

    num_human = len(rsms_per_dim_human)
    num_dnn = len(rsms_per_dim_dnn)
    duplicate_corrs = np.zeros(num_human)
    unique_corrs = np.zeros(num_dnn)
    used_dims = set()

    for i, rsm_human in enumerate(rsms_per_dim_human):
        print(f"Comparing human dimension {i}", end="\r")

        corrs_human_dim_i = np.zeros(num_dnn)

        for j, rsm_dnn in enumerate(rsms_per_dim_dnn):
            corr_ij = correlate_rsms(rsm_human, rsm_dnn)
            corrs_human_dim_i[j] = corr_ij

        sorted_dim_corrs = np.argsort(-corrs_human_dim_i)
        highest_corr = corrs_human_dim_i[sorted_dim_corrs[0]]
        duplicate_corrs[i] = highest_corr

        for dim in sorted_dim_corrs:
            if dim not in used_dims:
                used_dims.add(dim)
                highest_corr = corrs_human_dim_i[dim]
                unique_corrs[i] = highest_corr
                break

    if sort_by_corrs:
        duplicate_corrs = sort_by_corrs(duplicate_corrs)
        unique_corrs = sort_by_corrs(unique_corrs)

    return duplicate_corrs, unique_corrs


def plot_pairwise_rsm_corrs(unique_corrs, duplicate_corrs):
    # Create a DataFrame and reshape it to long format
    df = pd.DataFrame(
        {
            "Human Embedding RSM": range(len(unique_corrs)),
            "Unique": unique_corrs,
            "With Replacement": duplicate_corrs,
        }
    ).melt(
        "Human Embedding RSM",
        var_name="Pairing",
        value_name="Highest Pearson's r with DNN RSM",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = sns.color_palette()

    colors = [colors[2], colors[3]]
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
    xticks = [10, 20, 30, 40, 50, 60]
    ax.set(
        xlabel="Human Dimension RSM",
        ylabel="Pearson's r with DNN Dimension RSM",
        xticks=xticks,
        xticklabels=[str(x) for x in xticks],
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
