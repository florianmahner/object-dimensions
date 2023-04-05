#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" This is the main script for the human vs DNN comparison """

import os
from glob import glob
import toml
import matplotlib.pyplot as plt

import numpy as np
from object_dimensions.utils.utils import (
    load_sparse_codes,
    load_image_data,
    create_path_from_params,
    compute_rdm,
    fill_diag,
)
from object_dimensions.utils.utils import (
    load_sparse_codes,
    load_image_data,
    create_path_from_params,
    compute_rdm,
    fill_diag,
)
from experiments.human_dnn.jackknife import run_jackknife
from experiments.human_dnn.embedding_analysis import run_embedding_analysis
from object_dimensions import ExperimentParser
from scipy.stats import pearsonr, rankdata


parser = ExperimentParser(description="Analysis and comparison of embeddings")
parser.add_argument(
    "--dnn_path", type=str, help="Path to the base directory of the experiment runs"
)
parser.add_argument(
    "--human_path_base",
    type=str,
    help="Path to the base directory of the human embedding",
)
parser.add_argument(
    "--human_path_comp",
    type=str,
    help="Path to the base directory of the human embedding",
)
parser.add_argument(
    "--img_root",
    type=str,
    help="Path to the image root directory used for training the DNN. Contains behavior images and plus",
)
parser.add_argument(
    "--triplet_path",
    type=str,
    help="Path to the behavior triplets used for jackknife analysis",
)
parser.add_argument(
    "--evaluation_key",
    type=str,
    help="Key to aggregate over that is contained in the path",
)
parser.add_argument(
    "--analysis_key",
    type=str,
    help="Key to analyze that is contained in the parameters.npz file",
)



def get_rdm(embedding, method="correlation"):
    embedding = np.maximum(0, embedding)
    rsm = compute_rdm(embedding, method)
    rsm = fill_diag(rsm)
    rsm = rankdata(rsm).reshape(rsm.shape)

    return rsm


def normalise_rdm(rdm):
    rdm = rdm / np.max(rdm)
    return rdm


def compare_human_dnn(args):

    dnn_embedding, dnn_var = load_sparse_codes(args.dnn_path, with_var=True)
    human_embedding, human_var = load_sparse_codes(args.human_path_base, with_var=True)
    human_embedding_comp, human_var_comp = load_sparse_codes(args.human_path_comp, with_var=True)
    
    # # Load the image data
    plot_dir = create_path_from_params(args.dnn_path, "analyses", "human_dnn")
    print("Save all human dnn comparisons to {}".format(plot_dir))
    image_filenames, indices = load_image_data(args.img_root, filter_behavior=True)
    dnn_embedding = dnn_embedding[indices]
    dnn_var = dnn_var[indices]

    run_embedding_analysis(human_embedding, dnn_embedding, image_filenames, plot_dir)

    plot_dir = create_path_from_params(args.dnn_path, "analyses", "jackknife_human_dnn")

    
    run_jackknife(
        human_embedding,
        human_var,
        dnn_embedding,
        dnn_var,
        image_filenames,
        args.triplet_path,
        plot_dir,
        comparison="dnn",
    )

    # if args.human_path_comp:
    #     plot_dir = create_path_from_params(args.dnn_path, "analyses", "jacknife_human_human")
    #     run_jackknife(human_embedding, human_var, human_embedding_comp, human_var_comp, image_filenames, args.triplet_path, plot_dir, 12, "human")


if __name__ == "__main__":
    args = parser.parse_args()
    compare_human_dnn(args)
