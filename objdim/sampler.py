#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import argparse
import random
import os
import re

import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist

parser = argparse.ArgumentParser("Main tripletization script")
parser.add_argument("--in_path", type=str, help="Path to deep net features")
parser.add_argument("--out_path", type=str, help="Path to store Triplets")
parser.add_argument("--n_samples", type=int, help="Number of triplets to sample")
parser.add_argument(
    "--k",
    type=int,
    default=3,
    choices=[2, 3],
    help="Whether to sample pairs or triplets",
)
parser.add_argument(
    "--similarity",
    type=str,
    choices=["cosine", "dot", "euclidean"],
    help="Similarity function for pairwise sampling",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--adaptive", type=bool, default=False, help="If adaptive sampling or not"
)


class Sampler(object):
    def __init__(
        self,
        in_path,
        out_path,
        n_samples,
        k=3,
        train_fraction=0.9,
        seed=42,
        similarity="dot",
        behavior_triplet_path=None,
    ):
        self.in_path = in_path
        self.out_path = out_path
        self.n_samples = int(n_samples)
        self.k = k
        self.train_fraction = train_fraction
        self.seed = seed
        self.similarity = similarity
        self.behavior_triplet_path = behavior_triplet_path

    def load_domain(self):
        if not re.search(r"(npy)$", self.in_path):
            raise FileNotFoundError("Input file must be a .npy file")
        if not os.path.exists(self.out_path):
            print(f"\n....Creating output directory: {self.out_path}\n")
            os.makedirs(self.out_path)
        random.seed(self.seed)
        np.random.seed(self.seed)
        X = np.load(self.in_path)
        X = self.remove_nans_(X)
        X = self.remove_negatives_(X)  # positivity constraint also on vgg features!
        return X

    @staticmethod
    def remove_negatives_(X):
        return np.maximum(0, X)

    @staticmethod
    def remove_nans_(X):
        nan_indices = np.isnan(X[:, :]).any(axis=1)
        return X[~nan_indices]

    @staticmethod
    def softmax(z):
        proba = np.exp(z) / np.sum(np.exp(z))
        return proba

    @staticmethod
    def log_softmax_scaled(z, const):
        """see https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/"""
        z = z - const
        scaled_proba = np.exp(z) / np.sum(np.exp(z))
        scaled_log_proba = const + np.log(scaled_proba)

        return scaled_log_proba

    def get_choice(self, S, triplet):
        combs = list(itertools.combinations(triplet, 2))
        sims = [S[comb[0], comb[1]] for comb in combs]
        positive = combs[np.argmax(sims)]
        ooo = list(set(triplet).difference(set(positive)))
        choice = np.hstack((positive, ooo))

        return choice

    @staticmethod
    def random_choice(n_samples, combs):
        return combs[
            np.random.choice(np.arange(combs.shape[0]), size=n_samples, replace=False)
        ]

    @staticmethod
    def get_combinations(M, k):
        return np.array(list(itertools.combinations(range(M), k)))

    @staticmethod
    def cosine_matrix(X, a_min=-1.0, a_max=1.0):
        """Compute cosine-similarity matrix."""
        num = X @ X.T
        # compute vector l2-norm across rows
        l2_norms = np.linalg.norm(X, axis=1)
        denom = np.outer(l2_norms, l2_norms)
        cos_mat = (num / denom).clip(min=a_min, max=a_max)

        return cos_mat

    def sample_pairs(self, similarity="cosine"):
        X = self.load_domain(self.in_path)
        M = X.shape[0]
        if similarity == "cosine":
            S = self.cosine_matrix(X)
        elif similarity == "dot":
            S = X @ X.T
        elif similarity == "euclidean":
            S = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)

        else:
            raise NotImplementedError

        combs = self.get_combinations(M, self.k)
        random_sample = self.random_choice(self.n_samples, combs)
        return S, random_sample

    def random_combination(self, iterable, r):
        "Random selection from itertools.combinations(iterable, r)"
        pool = tuple(iterable)
        n = len(pool)
        # Sorting prevents adding duplicates!
        indices = sorted(random.sample(range(n), r))
        return tuple(pool[i] for i in indices)

    def sample_adaptive_similarity_judgements(self):
        """Create similarity judgements."""
        X = self.load_domain()
        M = X.shape[0]
        S = self._get_similarity_matrx(X)

        # Adaptive sampling of unique triplets
        unique_triplets = set()
        count = Counter()
        count.update({x: 0 for x in range(M)})

        # At the start all classes have zero counts and we sample uniformly
        p_per_item = [1 / M for _ in range(M)]
        sample_idx, n_iter = 1, 1
        while sample_idx < self.n_samples + 1:
            n_iter += 1
            print(
                f"{n_iter} samples drawn, {sample_idx}/{self.n_samples} added", end="\r"
            )
            triplet = np.random.choice(range(M), 3, replace=False, p=p_per_item)

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

        triplets = np.zeros((self.n_samples, self.k), dtype=int)
        for i, triplet in enumerate(unique_triplets):
            print(f"Process {i}/{self.n_samples} triplets", end="\r")
            choice = self.get_choice(S, triplet)
            triplets[i] = (
                choice  # probably returns a list of indices of shape k where for that image the odd one out is
            )
        return triplets

    def _get_similarity_matrx(self, X):
        if self.similarity == "cosine":
            S = self.cosine_matrix(X)
        elif self.similarity == "dot":
            S = X @ X.T
        elif self.similarity == "euclidean":
            D = cdist(X, X, "euclidean")
            S = 1 / (1 + D)  # transform a distance into a similarity measure
        else:
            raise NotImplementedError
        return S

    def sample_similarity_judgements(self):
        """Create similarity judgements."""
        X = self.load_domain()
        M = X.shape[0]
        S = self._get_similarity_matrx(X)

        if self.behavior_triplet_path:
            # Load triplets from behavior
            unique_triplets = np.loadtxt(self.behavior_triplet_path)
            unique_triplets = unique_triplets.astype(int)
            unique_triplets.sort(axis=1)

        else:
            unique_triplets = set()
            items = list(range(M))
            n_tri = len(unique_triplets)
            while n_tri < self.n_samples:
                print(
                    f"{n_tri} samples drawn, {n_tri}/{self.n_samples} added", end="\r"
                )
                sample = self.random_combination(items, 3)

                unique_triplets.add(sample)
                n_tri = len(unique_triplets)

        n_samples = len(unique_triplets)
        triplets = np.zeros((n_samples, self.k), dtype=int)
        for i, triplet in enumerate(unique_triplets):
            print(f"Process {i}/{n_samples} triplets", end="\r")
            choice = self.get_choice(S, triplet)
            triplets[i] = (
                choice  # probably returns a list of indices of shape k where for that image the odd one out is
            )
        return triplets

    def create_train_test_split(self, similarity_judgements):
        """Split triplet data into train and test splits."""
        random.seed(0)
        random.shuffle(similarity_judgements)
        N = similarity_judgements.shape[0]
        frac = int(N * self.train_fraction)
        train_split = similarity_judgements[:frac]
        test_split = similarity_judgements[frac:]

        N = similarity_judgements.shape[0]
        rnd_perm = np.random.permutation(N)
        train_split = similarity_judgements[
            rnd_perm[: int(len(rnd_perm) * self.train_fraction)]
        ]
        test_split = similarity_judgements[
            rnd_perm[int(len(rnd_perm) * self.train_fraction) :]
        ]
        return train_split, test_split

    def save_similarity_judgements(self, similarity_judgements):
        train_split, test_split = self.create_train_test_split(similarity_judgements)
        with open(os.path.join(self.out_path, "train_90.npy"), "wb") as train_file:
            np.save(train_file, train_split)
        with open(os.path.join(self.out_path, "test_10.npy"), "wb") as test_file:
            np.save(test_file, test_split)

    def run_and_save_tripletization(self, adaptive=False):
        if adaptive:
            similarity_judgements = self.sample_adaptive_similarity_judgements()
        else:
            similarity_judgements = self.sample_similarity_judgements()
        self.save_similarity_judgements(similarity_judgements)

    def run_and_save_pairwise(self, similarity="dot"):
        S, random_sample = self.sample_pairs(args.similarity)
        self.save_similarity_judgements(random_sample)
        with open(
            os.path.join(args.out_path, f"similarity_matrix_{similarity}.npy"), "wb"
        ) as f:
            np.save(f, S)

    def run(self):
        """Run sampler."""
        if self.k == 3:
            self.run_and_save_tripletization()
        else:
            self.run_and_save_pairwise()


if __name__ == "__main__":
    args = parser.parse_args()
    sampler = Sampler(
        in_path=args.in_path,
        out_path=args.out_path,
        n_samples=args.n_samples,
        k=args.k,
        seed=args.seed,
        similarity=args.similarity,
    )
    sampler.run()
