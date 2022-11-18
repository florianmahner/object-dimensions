#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os
import numpy as np

def build_triplet_dataset(triplet_path, n_train=None, n_val=None):
    """ Build a triplet dataset from a triplet directory containing sample triplets for all objects """

    # Find all files ending .npy or .txt and containing train or val
    files = os.listdir(triplet_path)
    if len(files) == 0:
        raise ValueError("No training or test files found in {}".format(triplet_path))

    for file in files:
        if file.endswith(".npy") and "train" in file:
            train = np.load(os.path.join(triplet_path, file))
        elif file.endswith(".npy") and "test" in file:
            test = np.load(os.path.join(triplet_path, file))
        elif file.endswith(".txt") and "train" in file:
            train = np.loadtxt(os.path.join(triplet_path, file))
        elif file.endswith(".txt") and "test" in file:
            test = np.loadtxt(os.path.join(triplet_path, file))
            
    # maybe need to do train/val split here beforehand and test=test?
    train_dataset = TripletDataset(train, n_train)
    val_dataset = TripletDataset(test, n_val)

    return train_dataset, val_dataset

class TripletDataset(torch.utils.data.Dataset):
    ''' Sample triplet indices from the list combinations'''
    def __init__(self, triplet_indices, n_samples=None):
        super().__init__()
        self.triplet_indices = triplet_indices
        self.triplets_indices = self.triplet_indices.astype(int)
        self.n_indices = n_samples if n_samples else len(triplet_indices) 

    def __getitem__(self, idx):
        return self.triplet_indices[idx]

    def __len__(self):
        return self.n_indices 


