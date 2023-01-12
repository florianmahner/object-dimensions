#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os
import numpy as np

def build_triplet_dataset(triplet_path, n_train=None, n_val=None, device="cpu"):
    """ Build a triplet dataset from a triplet directory containing sample triplets for all objects """

    # Find all files ending .npy or .txt and containing train or val
    files = os.listdir(triplet_path)
    if len(files) == 0:
        raise ValueError("No training or test files found in {}".format(triplet_path))

    for file in files:
        if file.endswith(".npy") and "train_90" in file:
            train = np.load(os.path.join(triplet_path, file))
        elif file.endswith(".npy") and "test_10" in file:
            test = np.load(os.path.join(triplet_path, file))
        elif file.endswith(".txt") and "train_90" in file:
            train = np.loadtxt(os.path.join(triplet_path, file))
        elif file.endswith(".txt") and "test_10" in file:
            test = np.loadtxt(os.path.join(triplet_path, file))
            
    # maybe need to do train/val split here beforehand and test=test?
    train_dataset = TripletDataset(train, n_train, device=device)
    val_dataset = TripletDataset(test, n_val, device=device)

    # train_dataset = TensorDataloader(train, batch_size=16384, shuffle=True, random_seed=42, device=device)
    # val_dataset = TensorDataloader(test, batch_size=16384, shuffle=False, random_seed=42, device=device)

    return train_dataset, val_dataset

class TripletDataset(torch.utils.data.Dataset):
    ''' Sample triplet indices from the list combinations'''
    def __init__(self, triplet_indices, n_samples=None, device="cpu"):
        super().__init__()
        self.triplet_indices = triplet_indices
        self.triplets_indices = self.triplet_indices.astype(int)
        self.triplet_indices = torch.from_numpy(self.triplet_indices).to(device)
        self.triplet_indices = self.triplet_indices.type("torch.LongTensor")
        self.n_indices = n_samples if n_samples else len(triplet_indices)

    def __getitem__(self, idx):
        return self.triplet_indices[idx]

    def __len__(self):
        return self.n_indices 


class TensorDataloader:
    """ Write a class that extract 1x3 indices from a a large array and includes
    functionality from torch dataloader with shuffling and batching
    """
    
    def __init__(self, data, batch_size, shuffle=True, random_seed=42, device="cpu"):
            self.dataset = data
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.random_seed = random_seed
            self.n_samples = data.shape[0]
         
            # self.n_batches = self.n_samples // self.batch_size
            self._get_batches()
            self.dataset = torch.from_numpy(self.dataset).to(device)
            self.dataset = self.dataset.type("torch.LongTensor")


    def _get_batches(self):
        # Check if number of samples can be divided by batch size and discard the last batch
        if self.n_samples % self.batch_size != 0:
            print("Number of samples is not divisible by batch size. Discarding last batch.")
            self.n_samples = self.n_samples - (self.n_samples % self.batch_size)
            self.dataset = self.dataset[:self.n_samples]
            self.n_batches = self.n_samples // self.batch_size
    
    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.shuffle:
            np.random.seed(self.random_seed)
            indices = np.random.permutation(self.n_samples)
        else:
            indices = np.arange(self.n_samples)

        for i in range(self.n_batches):
            batch_indices = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield self.dataset[batch_indices]

