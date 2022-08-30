# -*- coding: utf-8 -*-
import argparse
from bdb import Breakpoint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description='Cross Correlate Two Embeddings')
parser.add_argument('--embedding_1', type=str, default='./weights', help='path to first embedding dimension') 
parser.add_argument('--embedding_2', type=str, default='./weights', help='path to second embedding dimension') 
parser.add_argument('--save', type=bool, default=True, help='store cross correlation as pickle') 


def preprocess_embedding(embedding):
    # positivity constraint on both embeddings 
    embedding = np.maximum(embedding, 0)
    # sort the dimensions decreasingly for both embeddings
    embedding = embedding[:, np.argsort(-np.linalg.norm(embedding, axis=0, ord=1))]
    return embedding


def cross_correlate(args):

    embedding_1 = np.loadtxt(args.embedding_1)
    embedding_2 = np.loadtxt(args.embedding_2)
    embedding_1 = preprocess_embedding(embedding_1)
    embedding_2 = preprocess_embedding(embedding_2)


    dim1 = embedding_1.shape[1]
    dim2 = embedding_2.shape[1]

    if dim2 < dim1:
        print("Embedding 2 smaller than 1, switch order for comparison")
        tmp = embedding_2
        embedding_2 = embedding_1
        embedding_1 = tmp
        dim1 = embedding_2.shape[1]
        dim2 = embedding_2.shape[1]

    print('Shape of embedding 1 ', embedding_1.shape)
    print('Shape of embedding 2', embedding_2.shape)
    assert embedding_1.shape[0] == embedding_2.shape[0], "Both embeddings must have been trained on the same number of objects"

    # We proceed as follows:
    # We take the max of the first embedding.
    # Iterate over the second one and find the embedding with the highest correlation
    # Remove this embedding from the second embedding 
    # Repeat, i.e. take the second highest from the first etc.

    embedding_1 = embedding_1.T
    embedding_2 = embedding_2.T

    cross_corrs = np.zeros(dim1)
    all_corrs = np.zeros((dim1, dim2))

    for i, w_i in enumerate(embedding_1):

        largest_corr = 0
        largest_idx = np.inf

        for j, w_j in enumerate(embedding_2):

            corr_ij = np.corrcoef(w_i, w_j)[0, 1]
            all_corrs[i,j] = corr_ij

            if corr_ij > largest_corr:
                largest_corr = corr_ij
                largest_idx = j


        embedding_2 = np.delete(embedding_2, largest_idx, axis=0)
        cross_corrs[i] = largest_corr


    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    sns.set_context("notebook")
    sns.set_style("whitegrid")
    sns.lineplot(x=range(len(cross_corrs)), y=cross_corrs, color='black', ax=ax[0])
    ax[0].set_xlabel('Embedding Dimension')
    ax[0].set_ylabel("Pearson r")
    ax[0].set_title('Largest Correlating Dimensions')

    sns.heatmap(all_corrs, ax=ax[1], cmap='BuPu', cbar=False)
    # ax[1].imshow(all_corrs, cmap="BuPu", interpolation='nearest')
    ax[1].set_title("Cross Correlation Matrix")

    fig.tight_layout()
    # plt.savefig("cross_correlation_spose_spose.png", dpi=300)
    plt.savefig("../plots/cross_correlation_spose_vice_triplets_behavior_8096bs.png", dpi=300)


if __name__ == "__main__":

    args = parser.parse_args()
    # args.embedding_1 = "../spose_adaptive_lambda_005/weights_sorted_epoch0175.txt"
    # args.embedding_2 = "../spose_adaptive_lambda_004/weights_sorted_epoch0175.txt"
    # args.embedding_1 = "../learned_embeddings/spose_adaptive_lambda_004/weights_sorted_epoch0175.txt"
    # args.embedding_2 = "../weights_triplets_50mio/params/pruned_q_mu_epoch_500.txt"

    args.embedding_1 = "../spose_embedding_66d_sorted.txt"
    args.embedding_2 = "../weights_things_behavior_8196bs_adaptive_half/params/pruned_q_mu_epoch_5000.txt"
    # args.embedding_2 = "../weights_things_behavior_256bs/params/pruned_q_mu_epoch_500.txt"
    # args.embedding_2 = "../weights_things_behavior_8196bs_adaptive_complex/params/pruned_q_mu_epoch_200.txt"

    cross_correlate(args)