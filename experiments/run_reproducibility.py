""" Calculates the dimension and model reproducibility of representational embeddings 
across different random seeds """

import os
import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from argparse import ArgumentParser
from tqdm import tqdm
from typing import Dict, List, Tuple
import seaborn as sns

from objdim.utils import vectorized_pearsonr


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./results/behavior",
        help="Base directory of the experiment",
    )

    parser.add_argument(
        "--modality",
        type=str,
        default="behavior",
        choices=("behavior", "dnn", "voices"),
    )

    parser.add_argument("--run_analysis", action="store_true")
    return parser.parse_args()


def split_half_reliability(embeddings: np.ndarray, identifiers: str) -> Dict[str, list]:
    """
    Compute the split-half reliability of each dimension for each model.

    The method is as follows:
    1. Split the data objects in half using an odd and even mask.
    2. For a given model run, iterate across all dimensions $i$:
        - Identify the dimension in all other models $k$ that has the highest correlation with dimension $i$,
          calculated using the odd-masked data.
        - For model $k$, correlate this identified dimension with dimension $i$ using the even-masked data.
    This process results in a sampling distribution of Pearson r coefficients across all other model seeds.
    The sampling distribution of Pearson r is then transformed using Fisher-z so that it becomes z-scored (i.e.,
    normally distributed). The mean of this sampling distribution is taken as the average z-transformed
    reliability score. Finally, this score is inverted to get the average Pearson r reliability score.

    Parameters
    ----------
    embeddings : np.ndarray
        Embeddings in the shape of (n_embeddings, n_objects, n_dimensions).

    identifiers : str
        Identifiers of the embeddings, which could be model names or seeds.

    Returns
    -------
    dict
        A dictionary with identifiers as keys and Pearson r values across all dimensions.


    TODO - Take the pruned dimensions to calc fisher z!
    """
    assert (
        embeddings.ndim == 3
    ), "Embeddings must be 3-dimensional (n_embeddings, n_objects, n_dimensions)"
    n_embeddings, n_objects, n_dimensions = embeddings.shape

    np.random.seed(42)
    odd_mask = np.random.choice([True, False], size=n_objects)
    even_mask = np.invert(odd_mask)
    reliability_per_dim = defaultdict(list)

    for i in tqdm(range(n_embeddings)):
        ident = identifiers[i]
        reproducibility_across_embeddings = np.zeros((n_embeddings, n_dimensions))

        for j in range(n_embeddings):
            if i == j:
                continue

            emb_i = embeddings[i]
            emb_j = embeddings[j]

            # For each dim i, correlate with all other dims j
            corr_ij = vectorized_pearsonr(emb_i[odd_mask], emb_j[odd_mask])

            # For each dim i, find the dim j with the highest correlation
            highest_corrs = np.argmax(corr_ij, axis=1)

            # For each dim i, correlate with the dim j with the highest correlation across even objects
            even_corrs = np.zeros(n_dimensions)
            for k in range(n_dimensions):
                base_even = emb_i[even_mask][:, k]
                dim_match = highest_corrs[k]  # dim j with highest corr to dim i
                comp_even = emb_j[even_mask][:, dim_match]
                even_corrs[k] = vectorized_pearsonr(base_even, comp_even)

            reproducibility_across_embeddings[j] = even_corrs

        z_transformed = np.arctanh(reproducibility_across_embeddings)
        average = np.mean(z_transformed, axis=0)
        back_transformed = np.tanh(average)
        reliability_per_dim[ident] = back_transformed

    return reliability_per_dim


def find_mean_reliability(
    reliability_per_dim: Dict[str, list], n_pruned: Dict[str, int]
) -> Dict[str, float]:
    mean_reliabilities = {}
    for key, values in reliability_per_dim.items():
        ndim = n_pruned[key]
        values = values[:ndim]
        mean_reliabilities[key] = np.mean(values)

    return mean_reliabilities


def _preprocess_embedding(q_mu: np.ndarray) -> np.ndarray:
    q_mu = np.maximum(0, q_mu)
    ind = np.argsort(-q_mu.sum(0))
    q_mu = q_mu[:, ind]
    return q_mu


def _load_files(base_dir: str) -> Tuple[List[str], List[int]]:
    files = glob.glob(os.path.join(base_dir, "**/parameters.npz"), recursive=True)
    assert len(files) > 0, "No files found in {}".format(base_dir)
    files = sorted(files, key=lambda x: int(x.split("/")[-3]))
    seeds = np.array([int(f.split("/")[-3]) for f in files])
    return files, seeds


def plot_reliability(
    reliability_per_dim: np.ndarray,
    n_pruned: int,
    n_embeddings: int,
    out_path: str,
    modality: str,
) -> None:
    """Plot the reliability of each dimension across all models."""
    ndim = len(reliability_per_dim)
    fig, ax = plt.subplots(figsize=(6, 4))

    # Use seaborn to plot the line with improved aesthetics
    sns.lineplot(
        x=range(ndim),
        y=reliability_per_dim,
        color="#4C72B0",  # A more visually appealing blue from the default seaborn palette
        linewidth=2,
        ax=ax,
    )

    ax.set_xlabel("Dimension number", fontsize=10)
    ax.set_ylabel("Reproducibility of dimensions\n(Pearson Correlation)", fontsize=10)

    # Customize colors
    red = sns.color_palette("deep")[3]

    # Add vertical line for pruned dimensions with improved styling
    ax.axvline(
        x=n_pruned,
        color=red,
        linestyle="--",
        linewidth=2,
        label="Dimensions pruned",
    )

    sns.despine()
    ax.legend(fontsize=10, frameon=True, facecolor="white", edgecolor="lightgray")

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            "./results",
            "plots",
            f"reproducibility_across_dimensions_{modality}_classifier.4.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_reliability_across_seeds(
    mean_reliabilities: Dict[str, float],
    out_path: str,
    modality: str,
) -> None:

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.barplot(
        x=list(mean_reliabilities.keys()),
        y=list(mean_reliabilities.values()),
        edgecolor=".1",
        facecolor=".85",
        ax=ax,
    )
    ax.set_xlabel("Random Seed")
    ax.set_ylabel("Mean reproducibility (Pearson Correlation)")

    # make patches of lines inside the bar
    # for i, bar in enumerate(ax.patches):
    #     bar.set_hatch("//")
    #     bar.set_edgecolor("black")

    # Reduce the number of x-axis ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(11))

    # Reduce the number of y-axis ticks
    ax.yaxis.set_major_locator(plt.MaxNLocator(11))

    sns.despine(left=True, bottom=True)
    plt.savefig(
        os.path.join(
            "./results",
            "plots",
            f"reproducibility_across_seeds_{modality}_classifier.4.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_analysis(
    base_dir: str,
    modality: str,
    run_analysis: bool = True,
) -> None:
    out_path = os.path.join(
        "./results", "reliability_across_seeds_{}.pkl".format(modality)
    )

    files, seeds = _load_files(base_dir)
    embeddings, pruned_dims = defaultdict(list), defaultdict(list)
    for file, seed in zip(files, seeds):
        params = np.load(file)
        if params["method"] == "variational":
            q_mu = params["q_mu"]
        else:
            q_mu = params["weights"]
        q_mu = _preprocess_embedding(q_mu)

        embeddings[seed] = q_mu
        pruned_dims[seed] = params["dim_over_time"][-1]

    if run_analysis:
        identifiers = np.array(list(embeddings.keys()))
        embedding_array = np.array(list(embeddings.values()))

        # Find the key with the highest mean reliability
        reliability_per_dim = split_half_reliability(embedding_array, identifiers)
        mean_reliabilities = find_mean_reliability(reliability_per_dim, pruned_dims)

        best_seed = np.argmax(list(mean_reliabilities.values()))
        best_embedding = embeddings[best_seed]
        n_pruned = pruned_dims[best_seed]
        best_reliability_per_dim = reliability_per_dim[best_seed]

        out_dict = dict(
            reliability_per_dim=reliability_per_dim,
            best_reliability_per_dim=best_reliability_per_dim,
            best_embedding=best_embedding,
            best_seed=best_seed,
            n_pruned=n_pruned,
            mean_reliabilities=mean_reliabilities,
        )

        with open(out_path, "wb") as f:
            pickle.dump(out_dict, f)

    save = pickle.load(open(out_path, "rb"))
    best_reliability_per_dim = save["best_reliability_per_dim"]
    n_pruned = save["n_pruned"]
    best_seed = save["best_seed"]
    print("Best embedding is for seed {}".format(best_seed))
    n_embeddings = len(embeddings)
    best_reliability_per_dim = np.squeeze(best_reliability_per_dim)

    plot_reliability(
        best_reliability_per_dim,
        n_pruned,
        n_embeddings,
        out_path,
        modality,
    )

    plot_reliability_across_seeds(
        save["mean_reliabilities"],
        out_path,
        modality,
    )


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args.base_dir, args.modality, args.run_analysis)
