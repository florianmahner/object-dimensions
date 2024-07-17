#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import random
import itertools

import numpy as np

from pathlib import Path
from tomlparse import argparse
from collections import Counter
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from typing import Union, Callable, Optional, Iterable

Array = np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract features and tripletize from a dataset using a pretrained model and module"
    )
    parser.add_argument(
        "--feature_path",
        type=str,
        default="./data/image_data/things",
        help="Path to features to use for tripletization",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./data/triplets",
        help="Path to save the triplets",
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
        "--triplet_path",
        type=str,
        default=None,
        help="Path to any triplets to use for sampling, i.e. compute ooo. based on these triplets",
    )
    parser.add_argument(
        "--similarity",
        type=str,
        default="dot",
        choices=["dot", "cosine", "euclidean"],
        help="Similarity metric to use for tripletization",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of nearest neighbors to use. 3 = triplet, 2 pairwise",
        choices=[2, 3],
    )

    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.9,
        help="Fraction of data to use for training",
    )
    return parser.parse_args()


def cosine_matrix(X: Array, a_min: float = -1.0, a_max: float = 1.0):
    """Compute cosine-similarity matrix."""
    num = X @ X.T
    l2_norms = np.linalg.norm(X, axis=1)
    denom = np.outer(l2_norms, l2_norms) + eps
    S = (num / denom).clip(min=a_min, max=a_max)
    return S


def dot_matrix(X: Array) -> Array:
    """Compute dot-product matrix."""
    S = X @ X.T
    return S


def euclidean_matrix(X: Array) -> Array:
    """Compute euclidean similarity matrix."""
    D = cdist(X, X, "euclidean")
    S = 1 / (1 + D)
    return S


def get_similarity(X: Array, similarity: Union[str, Callable] = "dot") -> Array:
    similarity_functions = {
        "cosine": cosine_matrix,
        "dot": dot_matrix,
        "euclidean": euclidean_matrix,
    }

    if callable(similarity):
        S = similarity(X)
    elif isinstance(similarity, str):
        try:
            S = similarity_functions[similarity](X)
        except KeyError:
            raise ValueError(f"Similarity metric {similarity} not supported")
    else:
        raise TypeError("The 'similarity' must be either a string or a function")
    return S


def load_domain(path: str) -> Array:
    """Load features from a file can either be a .npy or .txt file."""
    search = re.search(r"(npy|txt)$", path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist")
    if not search:
        raise ValueError("Input file must be a .npy or .txt file")
    func = np.load if re.search(r"(npy)$", path) else np.loadtxt
    domain = func(path)
    return domain


@dataclass(init=True, repr=True)
class Sampler(object):
    feature_path: str
    out_path: str
    n_samples: int
    k: int = 3
    train_fraction: float = 0.9
    seed: int = 42
    sample_type: str = "random"
    similarity: Union[str, Callable] = "dot"
    transforms: Optional[Callable] = None
    triplet_path: Optional[str] = None

    def __post_init__(self):
        if self.k not in [2, 3]:
            raise ValueError(
                "Only triplets (k=3) and pairwise (k=2) are supported at the moment"
            )
        if self.train_fraction > 1 or self.train_fraction < 0:
            raise ValueError("Train fraction must be between 0 and 1")
        if self.sample_type not in ["random", "adaptive"]:
            raise ValueError("Sample type must be either 'random' or 'adaptive'")
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        self.X = load_domain(self.feature_path)
        if self.transforms:
            self.X = self.transforms(self.X)
        else:
            self.X = self.default_transforms(self.X)
        self.S = get_similarity(self.X, similarity=self.similarity)
        self.n_objects, self.n_features = self.X.shape

    def default_transforms(self, X: Array) -> Array:
        X = np.maximum(0, X)
        nan_indices = np.isnan(X[:, :]).any(axis=1)
        X = X[~nan_indices]
        return X

    def softmax(self, z: Array) -> Array:
        proba = np.exp(z) / np.sum(np.exp(z))
        return proba

    def get_choice(self, S: Array, triplet: Array) -> Array:
        combs = list(itertools.combinations(triplet, 2))
        sims = [S[comb[0], comb[1]] for comb in combs]
        positive = combs[np.argmax(sims)]
        ooo = list(set(triplet).difference(set(positive)))
        choice = np.hstack((positive, ooo))

        return choice

    def log_softmax_scaled(self, X: Array, const: float = 0.0) -> Array:
        """see https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/"""
        X = X - const
        scaled_proba = np.exp(X) / np.sum(np.exp(X))
        scaled_log_proba = const + np.log(scaled_proba)
        return scaled_log_proba

    def find_triplet_argmax(self, S: Array, triplet: Array) -> Array:
        combs = list(itertools.combinations(triplet, 2))
        sims = [S[comb[0], comb[1]] for comb in combs]
        positive = combs[np.argmax(sims)]
        ooo = list(set(triplet).difference(set(positive)))
        choice = np.hstack((positive, ooo))
        return choice

    def select_odd_one_outs(self, triplets: Iterable) -> Array:
        ooo = np.zeros((self.n_samples, self.k), dtype=int)
        for i, triplet in enumerate(triplets):
            ooo[i] = self.find_triplet_argmax(self.S, triplet)
        return ooo

    def sample_adaptive(self):
        """Create similarity judgements."""
        unique_triplets = set()
        count = Counter()
        count.update({x: 0 for x in range(self.n_objects)})

        # At the start all classes have zero counts and we sample uniformly
        p_per_item = [1 / self.n_objects for _ in range(self.n_objects)]
        sample_idx, n_iter = 1, 1
        while sample_idx < self.n_samples + 1:
            n_iter += 1
            print(
                f"{n_iter} samples drawn, {sample_idx}/{self.n_samples} added", end="\r"
            )
            triplet = np.random.choice(
                range(self.n_objects), 3, replace=False, p=p_per_item
            )

            # Using this we can avoid duplicate triplets when adding to the set
            triplet.sort()
            triplet = tuple(triplet)

            # Add to set and increase count if triplet is still unique
            if triplet not in unique_triplets:
                count.update(triplet)
                unique_triplets.add(triplet)
                sample_idx += 1

            # Update histogram of each class and sample random choices with the inverse of the actual distribution
            if sample_idx % 100_000 == 0:
                sum_count = sum(count.values())
                sorted_count = sorted(count.items())

                # Make smallest proba the largest
                inverse_probas_per_item = [1 - s[1] / sum_count for s in sorted_count]

                # Correct uniform distribution
                norm_probas = [
                    float(i) / sum(inverse_probas_per_item)
                    for i in inverse_probas_per_item
                ]
                p_per_item = norm_probas

        ooo_choices = self.select_odd_one_outs(unique_triplets)
        return ooo_choices

    def random_combination(self, iterable: Iterable, r: int):
        "Random selection from itertools.combinations(iterable, r)"
        pool = tuple(iterable)
        n = len(pool)
        indices = sorted(
            random.sample(range(n), r)
        )  # sorting prevents adding duplicates!
        return tuple(pool[i] for i in indices)

    def sample_random(self) -> Array:
        """Sample triplets based on the similarity matrix."""
        unique_triplets = set()
        items = list(range(self.n_objects))
        n_triplets = 0
        while n_triplets < self.n_samples:
            print(f"{n_triplets}/{self.n_samples} added", end="\r")
            sample = self.random_combination(items, 3)
            unique_triplets.add(sample)
            n_triplets = len(unique_triplets)

        ooo_choices = self.select_odd_one_outs(unique_triplets)
        return ooo_choices

    def sample_pairs(self) -> Array:
        combs = np.array(list(itertools.combinations(range(self.n_objects), self.k)))
        random_sample = combs[
            np.random.choice(
                np.arange(combs.shape[0]), size=self.n_samples, replace=False
            )
        ]
        return random_sample

    def train_test_split(self, ooo_choices: Union[list, Array]):
        """Split triplet data into train and test splits."""
        random.seed(0)
        np.random.shuffle(ooo_choices)
        N = ooo_choices.shape[0]
        frac = int(N * self.train_fraction)
        train_split = ooo_choices[:frac]
        test_split = ooo_choices[frac:]
        return train_split, test_split

    def run(self) -> None:
        self()

    def __call__(self) -> None:
        """Sample triplets and save them to disk."""
        if self.k == 2:
            choices = self.sample_pairs()

        # If the triplets are already provided, just load them and select the odd one out
        if self.triplet_path:
            unique_triplets = load_domain(self.triplet_path)
            unique_triplets = unique_triplets.astype(int)
            self.n_samples = unique_triplets.shape[0]
            choices = self.select_odd_one_outs(unique_triplets)
            fname = Path(self.triplet_path).stem + ".npy"
            with open(os.path.join(self.out_path, fname), "wb") as f:
                np.save(f, choices)
            return

        if self.sample_type == "adaptive":
            choices = self.sample_adaptive()
        else:
            choices = self.sample_random()

        train_split, test_split = self.train_test_split(choices)
        percentage = int(self.train_fraction * 100)
        with open(os.path.join(self.out_path, f"train_{percentage}.npy"), "wb") as f:
            np.save(f, train_split)
        with open(
            os.path.join(self.out_path, f"test_{100 - percentage}.npy"), "wb"
        ) as f:
            np.save(f, test_split)


if __name__ == "__main__":
    args = parse_args()

    sampler = Sampler(
        args.feature_path,
        args.out_path,
        n_samples=args.n_samples,
        k=args.k,
        train_fraction=args.train_fraction,
        seed=args.seed,
        similarity=args.similarity,
        triplet_path=args.triplet_path,
    )

    sampler.run()
