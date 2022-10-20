#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os
import numpy as np

def build_triplet_dataset(triplet_path, device):
    train = np.load(os.path.join(triplet_path, "train_90.npy"))
    test = np.load(os.path.join(triplet_path, "test_10.npy"))

    # maybe need to do train/val split here beforehand and test=test?
    train_dataset = TripletDataset(train, device=device)
    val_dataset = TripletDataset(test, device=device)

    return train_dataset, val_dataset

class TripletDataset(torch.utils.data.Dataset):
    ''' Sample triplet indices from the list combinations'''
    def __init__(self, triplet_indices, n_samples=None, device=torch.device("cpu")):
        super().__init__()
        self.triplet_indices = triplet_indices
        self.triplets_indices = self.triplet_indices.astype(int)
        # self.triplet_indices = torch.as_tensor(self.triplet_indices)
        # self.triplet_indices = self.triplet_indices.to(device)
        self.n_indices = n_samples if n_samples else len(triplet_indices) 

    def __getitem__(self, idx):
        return self.triplet_indices[idx]

    def __len__(self):
        return self.n_indices

