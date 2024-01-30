#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from object_dimensions.extraction.extract_model_features import extract_features
from object_dimensions import Sampler
from tomlparse import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract features and tripletize from a dataset using a pretrained model and module"
    )
    parser.add_argument(
        "--feature_path",
        type=str,
        default="./data/features",
        help="Path to feature matrix or residuals",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./data/triplets",
        help="Path to store triplets",
    )

    parser.add_argument(
        "--adaptive",
        action="store_true",
        default=False,
        help="Adaptively sample triplets",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=int(2e7),
        help="Number of samples to use for tripletization",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for tripletization"
    )
    parser.add_argument(
        "--behavior_triplet_path",
        type=str,
        default=None,
        help="Path to behavior triplets to use for tripletization",
    )
    parser.add_argument(
        "--similarity",
        type=str,
        default="dot",
        help="Similarity metric to use for tripletization",
    )
    return parser.parse_args()


def sample_triplets(
    n_samples,
    feature_path,
    out_path,
    adaptive=False,
    seed=0,
    similarity="dot",
    behavior_triplet_path=None,
):
    n_mio_samples = str(int(n_samples // 1e6)) + "mio"
    out_path = os.path.join(out_path, "triplets_{}".format(n_mio_samples))
    print("Start sampling triplets for the model...")
    sampler = Sampler(
        feature_path,
        out_path,
        n_samples=n_samples,
        k=3,
        train_fraction=0.9,
        seed=seed,
        behavior_triplet_path=behavior_triplet_path,
        similarity=similarity,
    )
    sampler.run_and_save_tripletization(adaptive)
    print("... Done!")


if __name__ == "__main__":
    args = parse_args()
    sample_triplets(
        args.n_samples,
        args.feature_path,
        args.out_path,
        args.adaptive,
        args.seed,
        args.similarity,
        args.behavior_triplet_path,
    )
