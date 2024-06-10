#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" This script computes all RSA analyses between human and DNN representations."""

import os
from object_dimensions.utils import (
    load_sparse_codes,
    load_image_data,
    load_deepnet_activations,
    load_concepts,
    create_path_from_params,
)

from experiments.rsa.pairwise import pairwise_rsm_comparison, plot_pairwise_rsm_corrs
from experiments.rsa.cumulative import run_cumulative_rsa, plot_cumulative_rsa
from experiments.rsa.global_analysis import global_rsa_analysis, plot_rsms

import matplotlib.pyplot as plt
from tomlparse import argparse


def load_filtered_data(
    image_root,
    feature_path,
    human_path,
    dnn_path,
):
    """Filter the DNN embedding with by behavior and return both embeddings and features."""
    dnn_embedding = load_sparse_codes(dnn_path, relu=True)
    human_embedding = load_sparse_codes(human_path, relu=True)
    _, indices = load_image_data(image_root, filter_behavior=True)
    features = load_deepnet_activations(feature_path, center=False, relu=True)
    features = features[indices]
    dnn_embedding = dnn_embedding[indices]
    return human_embedding, dnn_embedding, features


def run_pairwise(
    human_embedding,
    dnn_embedding,
    plot_dir,
):
    """Run pairwise RSA between human and DNN embeddings."""
    duplicates_rsa, unique_rsa = pairwise_rsm_comparison(
        human_embedding, dnn_embedding, sort_by_corrs=True
    )
    fig = plot_pairwise_rsm_corrs(unique_rsa, duplicates_rsa)
    fig.savefig(
        os.path.join(plot_dir, "human_dnn_pairwise_rsa.pdf"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def run_cumulative(
    human_embedding,
    dnn_embedding,
    plot_dir,
):
    """Run cumulative RSA between human and DNN embeddings."""
    cumulative_corrs = run_cumulative_rsa(human_embedding, dnn_embedding)
    fig = plot_cumulative_rsa(cumulative_corrs)
    fig.savefig(
        os.path.join(plot_dir, "cumulative_rsa.pdf"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def run_global(
    human_embedding,
    dnn_embedding,
    concepts,
    plot_dir,
):
    """Run global RSA between human and DNN embeddings."""
    rsm_human, rsm_dnn, global_corr = global_rsa_analysis(
        human_embedding,
        dnn_embedding,
        concepts,
    )
    plot_rsms(rsm_human, rsm_dnn, plot_dir)
    return global_corr


def run(
    img_root,
    human_embedding_path,
    dnn_embedding_path,
    feature_path,
    concept_path,
):
    concepts = load_concepts(concept_path)

    plot_dir = create_path_from_params(dnn_embedding_path, "rsa")
    human_embedding, dnn_embedding, features = load_filtered_data(
        img_root,
        feature_path,
        human_embedding_path,
        dnn_embedding_path,
    )

    global_corr = run_global(human_embedding, dnn_embedding, concepts, plot_dir)
    print(f"Global Pearson's r RSMs: Human-DNN: {global_corr}")
    print("Running Pairwise RSA...")
    # run_pairwise(human_embedding, dnn_embedding, plot_dir)
    print("Running Cumulative RSA...")
    run_cumulative(human_embedding, dnn_embedding, plot_dir)


if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Compare human and DNN performance on the same task."
        )
        parser.add_argument(
            "--human_path", type=str, help="Path to human embedding matrix."
        )
        parser.add_argument(
            "--dnn_path", type=str, help="Path to DNN embedding matrix."
        )
        parser.add_argument(
            "--img_root", type=str, help="Path to VGG feature matrix and filenames"
        )
        parser.add_argument(
            "--concept_path",
            type=str,
            default="./data/misc/category_mat_manual.tsv",
            help="Path to concept matrix",
        )
        parser.add_argument(
            "--corr_type",
            type=str,
            choices=["pearson", "spearman"],
            help="Type of correlation to use",
        )
        parser.add_argument("--run_analysis", action="store_true", help="Run analysis")
        parser.add_argument(
            "--feature_path",
            type=str,
            default="./features",
            help="Path to DNN features directory",
        )
        parser.add_argument(
            "--words48_path",
            type=str,
            default="./data/misc/words48.csv",
            help="Path to words48 file containing object categories used to fully sample things similarity matrix for noise ceilings",
        )
        parser.add_argument(
            "--human_rd_gt",
            type=str,
            help="Path to human RDM ground truth",
            default="./data/misc/rdm48_human.mat",
        )
        return parser.parse_args()

    args = parse_args()
    run(
        args.img_root,
        args.human_path,
        args.dnn_path,
        args.feature_path,
        args.concept_path,
    )
