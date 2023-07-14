#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from object_dimensions import ExperimentParser
import itertools
import random
import os
import re
import torch
from dataclasses import dataclass
from typing import Union, Callable, Tuple, List, Optional, Iterable
import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist


def parse_args():
    parser = ExperimentParser(
        description="Extract features and tripletize from a dataset using a pretrained model and module"
    )
    parser.add_argument(
        "--in_path",
        type=str,
        default="./data/image_data/things",
        help="Path to image dataset or any other dataset",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./data/triplets",
        help="Path to store image features",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vgg16_bn",
        help="Name of the model to use if we want to extrct features",
    )
    parser.add_argument(
        "--module_name",
        type=str,
        default="classifier.3",
        help="Name of the module to use",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        default=False,
        help="Extract features from the dataset",
    )
    parser.add_argument(
        "--tripletize",
        action="store_true",
        default=False,
        help="Tripletize the features",
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
        help="Similarity metric to use for tripletization",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for tripletization",
    )

    return parser.parse_args()


def default_transforms(X: np.ndarray) -> np.ndarray:
    X = np.maximum(0, X)
    nan_indices = np.isnan(X[:, :]).any(axis=1)
    X = X[~nan_indices]
    return X


def cosine_matrix(X: np.ndarray, a_min: float = -1.0, a_max: float = 1.0):
    """Compute cosine-similarity matrix."""
    num = X @ X.T
    l2_norms = np.linalg.norm(X, axis=1)
    denom = np.outer(l2_norms, l2_norms)
    S = (num / denom).clip(min=a_min, max=a_max)
    return S


def dot_matrix(X: np.ndarray) -> np.ndarray:
    """Compute dot-product matrix."""
    S = X @ X.T
    return S


def euclidean_matrix(X: np.ndarray) -> np.ndarray:
    """Compute euclidean similarity matrix."""
    D = cdist(X, X, "euclidean")
    S = 1 / (1 + D)
    return S


def get_similarity(
    X: np.ndarray, similarity: Union[str, Callable] = "dot"
) -> np.ndarray:
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


def load_domain(path: str) -> np.ndarray:
    """Load features from a file can either be a .npy or .txt file."""
    search = re.search(r"(npy|txt)$", path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist")
    if not search:
        raise ValueError("Input file must be a .npy or .txt file")
    func = np.load if re.search(r"(npy)$", path) else np.loadtxt
    domain = func(path)
    return domain


def load_model(model_name: str):
    from thingsvision import get_extractor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name in ["clip", "OpenCLIP"]:
        model = get_extractor(
            model_name="OpenCLIP",
            pretrained=True,
            device=device,
            source="custom",
            model_parameters={"variant": "ViT-H-14", "dataset": "laion2b_s32b_b79k"},
        )
    else:
        model = get_extractor(
            model_name, pretrained=True, device=device, source="torchvision"
        )
    return model


def extract_features(
    img_root: str,
    out_path: str,
    model_name: str = "vgg16_bn",
    module_name: str = "classifer.3",
    batch_size: int = 4,
):
    """Extract features from a dataset using a pretrained model"""
    from thingsvision.utils.storing import save_features
    from thingsvision.utils.data import DataLoader, ImageDataset
    from thingsvision.core.extraction import center_features

    extractor = load_model(model_name)
    dataset = ImageDataset(
        root=img_root,
        out_path=out_path,
        transforms=extractor.get_transformations(),
        backend=extractor.get_backend(),
    )
    assert len(dataset) > 0, "Dataset from path {} is empty!".format(img_root)

    filenames = dataset.images
    with open(out_path + "/filenames.txt", "w") as f:
        f.write("\n".join(filenames))

    batches = DataLoader(
        dataset=dataset, batch_size=batch_size, backend=extractor.get_backend()
    )

    features = extractor.extract_features(
        batches=batches, module_name=module_name, flatten_acts=True
    )

    if model_name in ["clip", "OpenCLIP"]:
        features = center_features(features)
    save_features(features, out_path, file_format="npy")


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
    transforms: Optional[Callable] = default_transforms  # Note make this optional
    triplet_path: Optional[str] = None

    def __post_init__(self):
        if self.k != 3:
            raise ValueError("Only triplets are supported at the moment")
        if self.train_fraction > 1 or self.train_fraction < 0:
            raise ValueError("Train fraction must be between 0 and 1")
        if self.sample_type not in ["random", "adaptive"]:
            raise ValueError("Sample type must be either 'random' or 'adaptive'")

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        self.X = load_domain(self.feature_path)
        if self.transforms:
            self.X = self.transforms(self.X)
        self.S = get_similarity(self.X, similarity=self.similarity)
        self.n_objects, self.n_features = self.X.shape

    def softmax(self, z: np.ndarray) -> np.ndarray:
        proba = np.exp(z) / np.sum(np.exp(z))
        return proba

    def get_choice(self, S: np.ndarray, triplet: np.ndarray) -> np.ndarray:
        combs = list(itertools.combinations(triplet, 2))
        sims = [S[comb[0], comb[1]] for comb in combs]
        positive = combs[np.argmax(sims)]
        ooo = list(set(triplet).difference(set(positive)))
        choice = np.hstack((positive, ooo))

        return choice

    def log_softmax_scaled(self, X: np.ndarray, const: float = 0.0) -> np.ndarray:
        """see https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/"""
        X = X - const
        scaled_proba = np.exp(X) / np.sum(np.exp(X))
        scaled_log_proba = const + np.log(scaled_proba)
        return scaled_log_proba

    def find_triplet_argmax(self, S: np.ndarray, triplet: np.ndarray) -> np.ndarray:
        combs = list(itertools.combinations(triplet, 2))
        sims = [S[comb[0], comb[1]] for comb in combs]
        positive = combs[np.argmax(sims)]
        ooo = list(set(triplet).difference(set(positive)))
        choice = np.hstack((positive, ooo))
        return choice

    def select_odd_one_outs(self, triplets: Iterable) -> np.ndarray:
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

    def sample_random(self) -> np.ndarray:
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

    def sample_pairs(self) -> np.ndarray:
        combs = np.array(list(itertools.combinations(range(self.n_objects), self.k)))
        random_sample = combs[
            np.random.choice(
                np.arange(combs.shape[0]), size=self.n_samples, replace=False
            )
        ]
        return random_sample

    def train_test_split(self, ooo_choices):
        """Split triplet data into train and test splits."""
        random.seed(0)
        random.shuffle(ooo_choices)
        N = ooo_choices.shape[0]
        frac = int(N * self.train_fraction)
        train_split = ooo_choices[:frac]
        test_split = ooo_choices[frac:]
        return train_split, test_split

    def __call__(self) -> None:
        """Sample triplets and save them to disk."""
        if self.k == 2:
            choices = self.sample_pairs()

        # If the triplets are already provided, just load them and select the odd one out
        if self.triplet_path:
            unique_triplets = load_domain(self.triplet_path)
            unique_triplets = unique_triplets.astype(int)
            unique_triplets.sort(axis=1)
            choices = self.select_odd_one_outs(unique_triplets)
            fname = os.path.basename(self.triplet_path)
            with open(os.path.join(self.out_path, fname), "wb") as f:
                np.save(f, choices)
            return

        if self.sample_type == "adaptive":
            choices = self.sample_adaptive()
        else:
            choices = self.sample_random()

        train_split, test_split = self.train_test_split(choices)
        percentage = self.train_fraction * 100
        with open(os.path.join(self.out_path, f"train_{percentage}.npy"), "wb") as f:
            np.save(f, train_split)
        with open(
            os.path.join(self.out_path, f"test_{100 - percentage}.npy"), "wb"
        ) as f:
            np.save(f, test_split)


if __name__ == "__main__":
    args = parse_args()
    if args.extract:
        extract_features(
            args.in_path,
            args.out_path,
            args.model_name,
            args.module_name,
            batch_size=args.batch_size,
        )

    feature_path = os.path.join(args.out_path, "features.npy")

    if args.tripletize:
        sampler = Sampler(
            feature_path,
            args.out_path,
            n_samples=args.n_samples,
            k=args.k,
            train_fraction=args.train_fraction,
            seed=args.seed,
            similarity=args.similarity,
            triplet_path=args.triplet_path,
        )
