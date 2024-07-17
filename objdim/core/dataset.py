#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os
import re
import numpy as np

from typing import Tuple, Optional
from torch.utils.data import Dataset


def get_triplet_dataset(
    triplet_path: str,
    n_train: Optional[int] = None,
    n_val: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[Dataset, Dataset]:
    """Build a triplet dataset from a triplet directory containing sample triplets for all objects"""
    # Find all files ending .npy or .txt and containing train or val
    files = os.listdir(triplet_path)
    if len(files) == 0:
        raise ValueError("No training or test files found in {}".format(triplet_path))

    files = [f for f in files if re.search(r"train_|test_", f)]
    files = [f for f in files if re.search(r"\.npy|\.txt", f)]
    files = sorted(files, key=lambda x: "test" in x)

    if len(files) == 0:
        raise ValueError(
            "No training file `train_x` or test files `test_1-x`found in {}, where x denotes the train/val split".format(
                triplet_path
            )
        )

    if len(files) != 2:
        raise ValueError("Found more than two files in {}".format(triplet_path))

    func = np.load if files[0].endswith(".npy") else np.loadtxt
    train = func(os.path.join(triplet_path, files[0]))
    test = func(os.path.join(triplet_path, files[1]))

    train_dataset = TripletDataset(train, n_train, device=device)
    val_dataset = TripletDataset(test, n_val, device=device)

    return train_dataset, val_dataset


class TripletDataset(Dataset):
    """Sample triplet indices from the list combinations"""

    def __init__(
        self,
        triplet_indices: np.ndarray,
        n_samples: Optional[int] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.triplet_indices = triplet_indices
        self.triplets_indices = self.triplet_indices.astype(int)
        self.triplet_indices = torch.from_numpy(self.triplet_indices).to(device)
        self.triplet_indices = self.triplet_indices.type("torch.LongTensor")
        self.n_indices = n_samples if n_samples else len(triplet_indices)
        self.n_objects = self.triplet_indices.max() + 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.triplet_indices[idx]

    def __len__(self) -> int:
        return self.n_indices
