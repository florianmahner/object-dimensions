# -*- coding: utf-8 -*-
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from object_dimensions.utils.utils import load_sparse_codes

from object_dimensions.analyses.human_dnn.embedding_analysis import compare_modalities


parser = argparse.ArgumentParser(description="Cross Correlate Two Embeddings")
parser.add_argument(
    "--embedding_1",
    type=str,
    default="./weights",
    help="path to first embedding dimension",
)
parser.add_argument(
    "--embedding_2",
    type=str,
    default="./weights",
    help="path to second embedding dimension",
)
parser.add_argument(
    "--save", type=bool, default=True, help="store cross correlation as pickle"
)


def cross_correlate(embedding_1, embedding_2, plot=True):

    if not isinstance(embedding_1, np.ndarray):
        embedding_1 = load_sparse_codes(embedding_1)
        embedding_2 = load_sparse_codes(embedding_2)

        # embedding_2 = np.load(embedding_2)

    dim1 = embedding_1.shape[1]
    dim2 = embedding_2.shape[1]

    if dim2 < dim1:
        print("Embedding 2 smaller than 1, switch order for comparison")
        tmp = embedding_2
        embedding_2 = embedding_1
        embedding_1 = tmp
        dim1 = embedding_1.shape[1]
        dim2 = embedding_2.shape[1]

    print("Shape of embedding 1 ", embedding_1.shape)
    print("Shape of embedding 2", embedding_2.shape)
    assert (
        embedding_1.shape[0] == embedding_2.shape[0]
    ), "Both embeddings must have been trained on the same number of objects"

    # We proceed as follows:
    # We take the max of the first embedding.
    # Iterate over the second one and find the embedding with the highest correlation
    # Remove this embedding from the second embedding
    # Repeat, i.e. take the second highest from the first etc.
    embedding_1 = embedding_1.T
    embedding_2 = embedding_2.T

    cross_corrs = np.zeros(dim1)
    embedding_1
    all_corrs = np.zeros((dim1, dim2))

    for i, w_i in enumerate(embedding_1):

        largest_corr = 0
        largest_idx = np.inf

        for j, w_j in enumerate(embedding_2):
            corr_ij = np.corrcoef(w_i, w_j)[0, 1]
            all_corrs[i, j] = corr_ij
            if corr_ij > largest_corr:
                largest_corr = corr_ij
                largest_idx = j

        # embedding_2 = np.delete(embedding_2, largest_idx, axis=0)
        cross_corrs[i] = largest_corr

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(6, 6))
        sns.set_context("notebook")
        sns.set_style("whitegrid")
        sns.lineplot(x=range(len(cross_corrs)), y=cross_corrs, color="black", ax=ax[0])
        ax[0].set_xlabel("Embedding Dimension")
        ax[0].set_ylabel("Pearson r")
        ax[0].set_title("Largest Correlating Dimensions")

        print(min(dim1, dim2))
        # ax[0].set_xlim([0, min(dim1, dim2)])

        sns.heatmap(all_corrs, ax=ax[1], cmap="BuPu", cbar=False)
        # ax[1].imshow(all_corrs, cmap="BuPu", interpolation='nearest')
        ax[1].set_title("Cross Correlation Matrix")
        ax[1].set_xlabel("VICE")
        ax[1].set_ylabel("SPose")

        fig.tight_layout()
        # plt.savefig("cross_correlation_spose_spose.png", dpi=300)
        plt.savefig(
            "./misc/plots/cross_correlation_spose_vice_triplets_behavior_sslab_256_new.png",
            dpi=300,
        )

    return cross_corrs


if __name__ == "__main__":
    args = parser.parse_args()
    args.embedding_1 = "./results/spose_embedding_66d_sorted.txt"
    args.embedding_2 = "./results/dummy/behavior/4.58mio/sslab/100/256/1.0/1.0/42/params/pruned_q_mu_epoch_2500.txt"

    # args.embedding_2 = "./results/4096_nll/behavior/4.58mio/sslab/100/4096/1.4/0.25/0/params/pruned_q_mu_epoch_2000.txt"

    # args.embedding_2 = "./results/behavior_256/behavior/4.58mio/sslab/100/256/1.0/0.25/0/params/pruned_q_mu_epoch_1500.txt"
    # args.embedding_2 = "./results/128bs_behavior/behavior/4.58mio/sslab/100/128/1.0/0.25/0/params/pruned_q_mu_epoch_600.txt"
    # args.embedding_2 = "./results/new_run_log_gauss_normal_prune/behavior/4.58mio/gauss/100/256/1.0/0.25/0/params/embedding_epoch_300.txt"

    # args.embedding_2 = "./final_embedding_lukas.npy"

    cross_correlate(args.embedding_1, args.embedding_2)
