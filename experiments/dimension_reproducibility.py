import numpy as np
from scipy.stats import pearsonr
import toml
import glob
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from collections import defaultdict
from object_dimensions import ExperimentParser

""" Calculates the reproducbility of dimensions usign odd and even objects, i.e. split half reliability?"""

parser = ExperimentParser()
parser.add_argument(
    "--base_dir",
    type=str,
    default="./results/behavior",
    help="Base directory of the experiment",
)


def reliability_per_dim(
    n_objects, n_dimensions, n_embeddings, embeddings, pruned_across_embedding
):
    """Calculates the reliability of each dimension across embeddings using odd and even objects and a
    split half reliability measure."""

    odd_mask = np.arange(n_objects) % 2 == 1
    even_mask = np.arange(n_objects) % 2 == 0

    reliability_per_dim = defaultdict(list)

    for e in range(n_embeddings):
        for i in range(n_dimensions):
            # this is the base embedding
            embed_i = embeddings[e][:, i]
            k_per_dim = []

            # Remove the embedding that is the same
            compared_embeddings = list(np.arange(n_embeddings))
            compared_embeddings.pop(e)
            for k in compared_embeddings:
                odd_corrs = []

                # For a compared embedding iterate across all domesions
                for j in range(n_embeddings):
                    embed_jk = embeddings[k][:, j]
                    odd_correlation = pearsonr(embed_i[odd_mask], embed_jk[odd_mask])[0]
                    odd_corrs.append(odd_correlation)

                # Take the clostest match and correlate with the even embedding
                closest_match = np.argmax(odd_corrs)
                embed_compare = embeddings[k][:, closest_match]
                even_corr = pearsonr(embed_i[even_mask], embed_compare[even_mask])[0]
                k_per_dim.append(even_corr)

            # Calculate the mean correlation across all compared embeddings using a fisher z transform
            # that transforms the correlation to a normal distribution
            z_transformed = np.arctanh(np.array(k_per_dim))
            average = np.mean(z_transformed)
            back_transformed = np.tanh(average)

            reliability_per_dim[str(e)].append(back_transformed)
        print(
            "Mean Reliability for embedding {} is {}".format(
                e, np.mean(reliability_per_dim[str(e)])
            )
        )

    # Get the best embedding with highest mean reliability
    best_embedding = np.argmax(
        [np.mean(reliability_per_dim[str(e)]) for e in range(n_embeddings)]
    )

    n_pruned = pruned_across_embedding[best_embedding]

    print("Best embedding is for seed {}".format(best_embedding))
    best_reliability_per_dim = reliability_per_dim[str(best_embedding)]
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
            "./results", "plots", "reproducibility_across_dimensions_spose_behavior.png"
        ),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )


def preprocess_embedding(q_mu):
    q_mu = np.maximum(0, q_mu)
    ind = np.argsort(-q_mu.sum(0))
    q_mu = q_mu[:, ind]
    return q_mu


def run_analysis(base_dir):
    files = glob.glob(base_dir + "**/parameters.npz", recursive=True)
    assert len(files) > 0, "No files found in {}".format(base_dir)
    cfg = glob.glob(base_dir + "**/config.toml", recursive=True)[0]
    cfg = toml.load(cfg)

    n_dimensions = cfg["init_dim"]
    n_embeddings = len(files)

    # Take one base model (original). Check reproducibility to all other models
    # Concatenate all embedfinhgs on one dimension?
    embeddings, pruned_dims = [], []
    for f in files:
        params = np.load(f)
        if params["method"] == "variational":
            q_mu = params["q_mu"]
        else:
            q_mu = params["weights"]
        q_mu = preprocess_embedding(q_mu)
        embeddings.append(q_mu)
        pruned_dims.append(params["dim_over_time"][-1])

    n_objects = q_mu.shape[0]
    reliability_per_dim(n_objects, n_dimensions, n_embeddings, embeddings, pruned_dims)


if __name__ == "__main__":
    args = parser.parse_args()
    run_analysis(args.base_dir)
