#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import os

import numpy as np

from deep_embeddings import VI
from deep_embeddings import SpikeSlabPrior, ExponentialPrior
from deep_embeddings import TripletDataset
from deep_embeddings import MLTrainer
from deep_embeddings import DeepEmbeddingLogger

parser = argparse.ArgumentParser()
parser.add_argument("--feature_path",type=str, help="Path to the feature file")
parser.add_argument("--triplet_path", type=str, help="Path to the triplet file")
parser.add_argument("--log_path", type=str, default="./results", help="Path to store all training outputs and model checkpoints")
parser.add_argument("--modality", type=str, default="deep", choices=("deep", "behavior"), help="Modality to train on")
parser.add_argument("--load_model", default=False, choices=("True", "False"), help="Load a pretrained model from log path")
parser.add_argument("--fresh", default=False, choices=("True", "False"), help="Train a new model and erase previous log dir")
parser.add_argument("--tensorboard", default=False, choices=("True", "False"), help="Use tensorboard to log training")
parser.add_argument("--init_dim", type=int, default=100, help="Initial dimensionality of the latent space")
parser.add_argument("--prior", type=str, default="sslab", choices=["sslab", "exp"], help="Prior to use")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--n_epochs", type=int, default=3000, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--stability_time", type=int, default=100, help="Number of epochs to train before checking stability")
parser.add_argument("--rnd_seed", type=int, default=42, help="Random seed")
parser.add_argument("--gamma", type=float, default=0.5, help="Gamma parameter to balance KL div and reconstruction loss")
parser.add_argument("--params_interval", type=int, default=100, help="Interval to save learned embeddings")
parser.add_argument("--checkpoint_interval", type=int, default=100, help="Interval to save model checkpoints")


def build_triplet_dataset(triplet_path, device):
    train = np.load(os.path.join(triplet_path, "train_90.npy"))
    test = np.load(os.path.join(triplet_path, "test_10.npy"))

    # maybe need to do train/val split here beforehand and test=test?
    train_dataset = TripletDataset(train, device=device)
    val_dataset = TripletDataset(test, device=device)

    return train_dataset, val_dataset

def train(args):
    assert (args.fresh and args.load_model) is not True, "You can either load a model (config.load_model) or train a new one (config.fresh), not both!"
    
    torch.manual_seed(args.rnd_seed)
    np.random.seed(args.rnd_seed)

    features = np.load(args.feature_path)
    n_objects = features.shape[0]

    if args.modality == "behavior":
        n_objects = 1854

    model = VI(n_objects, args.init_dim)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=[0,1,2])
    
    else:
        model = torch.nn.DataParallel(model, device_ids=[0])
    model.to(device)

    if args.prior == 'sslab':
        prior = SpikeSlabPrior(n_objects, args.init_dim, spike=0.25, slab=1.0, pi=0.5)
        # prior = SpikeSlabPriorConstrained(n_objects, args.init_dim, spike=0.25, slab=1.0, pi=0.5)
    elif args.prior == 'exp':
        prior = ExponentialPrior(n_objects, lmbda=1.0)
    else:
        raise NameError('Unknown prior')

    train_dataset, val_dataset = build_triplet_dataset(args.triplet_path, device)

    # If all data on GPU, num workers need to be 0 and pin memory false
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Build loggers and 
    logger = DeepEmbeddingLogger(model, args)

    trainer = MLTrainer(model, prior, train_loader, val_loader, logger, device)
    trainer.parse_from_config(args)

    trainer.train()


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)