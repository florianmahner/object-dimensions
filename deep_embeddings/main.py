#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import os

import numpy as np

from deep_embeddings import VI
from deep_embeddings import SpikeSlabPrior, ExponentialPrior, WeibullPrior
from deep_embeddings import build_triplet_dataset
from deep_embeddings import EmbeddingTrainer
from deep_embeddings import DeepEmbeddingLogger

parser = argparse.ArgumentParser()
parser.add_argument("--triplet_path", type=str, help="Path to the triplet file")
parser.add_argument("--data_path", type=str, default="./data/models/vgg16bn")
parser.add_argument("--model_name", type=str, default="vgg16_bn", help="Name of the model")
parser.add_argument("--module_name", type=str, default="classifier.3", help="Name of the module for which features have been extracted")
parser.add_argument("--log_path", type=str, default="./results", help="Path to store all training outputs and model checkpoints")
parser.add_argument("--modality", type=str, default="deep", choices=("deep", "behavior"), help="Modality to train on")
parser.add_argument("--fresh", default=False, action='store_true', help="Start clean and delete old path content")
parser.add_argument("--load_model", default=False, action='store_true', help="Load a pretrained model from log path")
parser.add_argument("--tensorboard", default=False, action='store_true', help="Use tensorboard to log training")
parser.add_argument("--init_dim", type=int, default=100, help="Initial dimensionality of the latent space")
parser.add_argument("--prior", type=str, default="sslab", choices=["sslab", "exp", "weibull"], help="Prior to use")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--n_epochs", type=int, default=3000, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--stability_time", type=int, default=100, help="Number of epochs to train before checking stability")
parser.add_argument("--rnd_seed", type=int, default=42, help="Random seed")
parser.add_argument("--gamma", type=float, default=0.5, help="Gamma parameter to balance KL div and reconstruction loss")
parser.add_argument("--params_interval", type=int, default=100, help="Interval to save learned embeddings")
parser.add_argument("--checkpoint_interval", type=int, default=100, help="Interval to save model checkpoints")
parser.add_argument("--spike", type=float, default=0.25, help="Spike parameter for spike and slab prior")
parser.add_argument("--slab", type=float, default=1.0, help="Slab parameter for spike and slab prior")
parser.add_argument("--pi", type=float, default=0.5, help="Pi parameter for spike and slab prior")
parser.add_argument("--identifier", type=str, default="", help="Identifier for the experiment")


def train(args):
    if args.fresh and args.load_model:
        raise ValueError("Cannot load a model and train from scratch at the same time")

    torch.manual_seed(args.rnd_seed)
    np.random.seed(args.rnd_seed)

    feature_path = os.path.join(args.data_path, args.module_name, "features.npy")
    features = np.load(feature_path)
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
    elif args.prior == 'exp':
        prior = ExponentialPrior(lmbda=1.0)
    elif args.prior == "weibull":
        prior = WeibullPrior(lmbda=1.0, k=1.5)
    else:
        raise NameError('Unknown prior')

    train_dataset, val_dataset = build_triplet_dataset(args.triplet_path, device)

    # If all data on GPU, num workers need to be 0 and pin memory false
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Create a nested out path giving the config arguments
    n_mio_samples = int((len(train_dataset) + len(val_dataset)) / 1e6)
    n_samples_iden = str(n_mio_samples) + "mio"

    # Join all arguments to create a unique log path
    model_type = os.path.basename(args.data_path)
    log_path = os.path.join(args.log_path, args.identifier, model_type,  n_samples_iden, 
                            args.modality, args.prior, str(args.init_dim), str(args.batch_size), str(args.gamma), str(args.rnd_seed))

    # Build loggers and 
    logger = DeepEmbeddingLogger(log_path, model, args)

    # Print the number of millions in sample size as string
    print("Training on {} million triplets".format(len(train_dataset) / 1e6))

    trainer = EmbeddingTrainer(model, prior, train_loader, val_loader, logger, device)
    trainer.parse_from_args(args)

    trainer.train()


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)