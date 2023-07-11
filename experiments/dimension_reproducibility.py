import os
import toml
import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from collections import defaultdict
from argparse import ArgumentParser
from tqdm import tqdm
from typing import Dict, Union, List, Tuple


""" Calculates the reproducbility of dimensions using odd and even objects, i.e. split half reliability"""

parser = ArgumentParser()
parser.add_argument(
    "--base_dir",
    type=str,
    default="./results/behavior",
    help="Base directory of the experiment",
)

parser.add_argument(
    "--modality", type=str, default="behavior", choices=("behavior", "dnn")
)


def vectorized_pearsonr(base: np.ndarray, comp: np.ndarray) -> Union[float, np.ndarray]:
    """Alterntive to scipy.stats.pearsonr that is vectorized over the first dimension for
    fast pairwise correlation calculation."""
    if base.shape != comp.shape:
        raise ValueError(
            "Input arrays must have the same dimensions; "
            f"base.shape = {base.shape}, comp.shape = {comp.shape}"
        )
    if base.ndim < 2:
        base = base[:, None]
    if comp.ndim < 2:
        comp = comp[:, None]
    n = base.shape[1]
    covariance = np.cov(base.T, comp.T, ddof=1)
    base_std = np.sqrt(covariance[:n, :n].diagonal())
    comp_std = np.sqrt(covariance[n:, n:].diagonal())
    pearson_r = covariance[:n, n:] / np.outer(base_std, comp_std)
    return pearson_r


def split_half_reliability(embeddings: np.ndarray, identifiers: str) -> Dict[str, list]:
    """For two observers do:
    1. Split the data objects in half using an odd and even mask
    2. For a model run, iterate across all dimensions $i$
        - Find the highest correlating dimension across all other models $k$ and all other dimensions.
        Calculate this using the odd-mask
        - For model $k$ correlate the maximum correlation for that dimension $i$ using the even mask
    We then obtain a *sampling distribution* of Pearson r coefficients across all other model seeds.
    We Fisher-Z transform this sampling distribution of Pearson r so that they become z-scored (i.e. normally distributed)
    We then take the mean of that sampling distribution to have the average z-transformed reliability score
    We invert this to get the average Pearson r reliability score
    """
    n_embeddings, n_objects, n_dimensions = embeddings.shape
    assert n_embeddings == len(
        identifiers
    ), "Number of embeddings and identifiers must match"

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
        reliability_per_dim[ident].append(back_transformed)

    return reliability_per_dim


def find_mean_reliability(
    reliability_per_dim: Dict[str, list], n_pruned: Dict[str, int]
) -> Dict[str, float]:
    mean_reliabilities = {}
    for key, values in reliability_per_dim.items():
        ndim = n_pruned[key]
        values = values[:ndim]
        mean_reliability = np.mean(values)
        mean_reliabilities[key] = mean_reliability
        print("Mean reliability for seed {} is {}".format(key, mean_reliability))

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


def run_analysis(base_dir: str, modality: str) -> None:
    files, seeds = _load_files(base_dir)
    embeddings, pruned_dims = [], []
    for file in files:
        params = np.load(file)
        if params["method"] == "variational":
            q_mu = params["q_mu"]
        else:
            q_mu = params["weights"]
        q_mu = _preprocess_embedding(q_mu)

        embeddings.append(q_mu)
        pruned_dims.append(params["dim_over_time"][-1])

    embeddings = np.array(embeddings)
    reliability_per_dim = split_half_reliability(embeddings, seeds)
    mean_reliabilities = find_mean_reliability(reliability_per_dim, pruned_dims)

    best_embedding = np.argmax(mean_reliabilities)
    best_embedding = embeddings[best_embedding]
    n_pruned = pruned_dims[best_embedding]
    best_seed = seeds[best_embedding]

    print("Best embedding is for seed {}".format(best_seed))
    best_reliability_per_dim = reliability_per_dim[best_seed]

    out_dict = dict(
        reliability_per_dim=reliability_per_dim,
        best_reliability_per_dim=best_reliability_per_dim,
        best_seed=best_seed,
        mean_reliabilities=mean_reliabilities,
    )

    out_path = os.path.join(
        "./results", "reliability_across_seeds_{}.pkl".format(modality)
    )
    with open(out_path, "wb") as f:
        pickle.dump(out_dict, f)

    n_embeddings, _, n_dimensions = embeddings.shape
    data = pd.DataFrame(
        {
            "Dimension": np.arange(n_dimensions),
            "Reproducibility": best_reliability_per_dim,
        }
    )
    ax = sns.lineplot(x="Dimension", y="Reproducibility", data=data, color="black")
    ax.set_xlabel("Dimension number")
    ax.set_ylabel("Reproducibility of dimensions (Pearson's r)".format(n_embeddings))
    ax.set_title("N = {} runs, Batch Size = 256".format(n_embeddings))
    ax.axvline(x=n_pruned, color="red", linestyle="--")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            "./results", "plots", f"reproducibility_across_dimensions_{modality}.png"
        ),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()


if __name__ == "__main__":
    args = parser.parse_args()
    run_analysis(args.base_dir, args.modality)
