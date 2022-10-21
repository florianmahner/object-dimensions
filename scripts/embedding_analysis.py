#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# We want to:
# (i) agggregate across different models or seeds
# (ii) compare different models or seeds with 
#       (a) the plotting of the rdms 
#       (b) correlating the rdms with each other!
# (iii) do ablation stuff on the rdms / correlate rdms with each other!
# (iv) do the human vs rdm comparisom stuff also for different models / seeds

# saving all rdms in the analysis folder

import argparse
import os
import itertools
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from evaluate_across_runs import aggregate_params, aggregate_results
from deep_embeddings.utils.utils import transform_weights, correlate_rsms, compute_rsm, filter_embedding_by_behavior
from deep_embeddings.analyses.human_dnn.compare_human_dnn import compare_modalities

parser = argparse.ArgumentParser(description='Analysis and comparison of embeddings')
parser.add_argument("--dnn_base_path", type=str, help="Path to the base directory of the experiment runs")
parser.add_argument("--human_path", type=str, help="Path to the base directory of the human embedding")
parser.add_argument("--file_names", type=str, help="Txt file directory of all image paths")
parser.add_argument("--evaluation key", type=str, help="Key to aggregate over that is contained in the path")


def collect_embeddings(dnn_path, human_path, evaluation_key="models"):
    """ We aggregate across different models in the same directory and then try to correlate """
    human_params = aggregate_params(human_path)
    
    # dnn_params = aggregate_params(dnn_path, evaluation_key=None)
    dnn_params = aggregate_params(dnn_path)

    embedding_dict = defaultdict(dict)
    embedding_dict["human"]["q_mu"]= human_params["pruned_q_mu"]
    embedding_dict["human"]["q_var"]= human_params["pruned_q_var"]

    for key, val in dnn_params.items():
        embedding_dict[key]["q_mu"] = val["pruned_q_mu"]
        embedding_dict[key]["q_var"] = val["pruned_q_var"]

    # Ensure positivity constraint and sort the means of an embedding
    for key, val in embedding_dict.items():
        if key == "q_mu":
            embedding_dict[key] = transform_weights(val["q_mu"])

    return embedding_dict


def rsm_analyse_embeddings(embedding_dict, dnn_path, file_names):
    """ We plot the embeddings for different models or seeds """
    keys = embedding_dict.keys()
    key_combinations = itertools.combinations(keys, 2)
    file_names = np.loadtxt(file_names, dtype=str)

    for key_1, key_2 in key_combinations:
        embedding_1 = embedding_dict[key_1]["q_mu"]
        embedding_2 = embedding_dict[key_2]["q_mu"]
        
        # Limit the number of images to the ones obtained from behavior, if one embedding is from behavior
        if embedding_1.shape[0] > 1854 and embedding_2.shape[0] == 1854:
            embedding_1, _ = filter_embedding_by_behavior(embedding_1, file_names)

        if embedding_1.shape[0] == 1854 and embedding_2.shape[0] > 1854:
            embedding_2, _ = filter_embedding_by_behavior(embedding_2, file_names)

        # Get the RSMs
        rsm_1 = compute_rsm(embedding_1)
        rsm_2 = compute_rsm(embedding_2)

        # Correlate the RSMs
        rho = correlate_rsms(rsm_1, rsm_2)
        print("Correlation between {} and {} is {}".format(key_1, key_2, rho))

        embedding_dict[key_1]["rsm"] = rsm_1
        embedding_dict[key_2]["rsm"] = rsm_2


        # Correlate dimensions and plot!
        dim_corrs = compare_modalities(embedding_1, embedding_2, duplicates=False)[2]
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.plot(range(len(dim_corrs)), dim_corrs, color='black', linewidth=2)
        ax.set_xlabel("Latent dimension")
        ax.set_ylabel("Pearson r")
        ax.set_title("{} vs {}".format(key_1, key_2))
        plt.show()
        plt.tight_layout()

        fig.savefig(os.path.join(dnn_path, "dim_corrs_{}_{}.png".format(key_1, key_2)))
        
    
    return embedding_dict

if __name__ == '__main__':

    args = parser.parse_args()
    # args.dnn_base_dir = "./results/exp"
    args.dnn_base_dir = "./results/50mio"
    args.human_path = "./results/behavior"
    args.file_names = "./data/models/vgg16_bn/classifier.3/file_names.txt"

    embedding_dict = collect_embeddings(args.dnn_base_dir, args.human_path, "model")
    rsm_analyse_embeddings(embedding_dict, args.dnn_base_dir, args.file_names)









