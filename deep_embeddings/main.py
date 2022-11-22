#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import os

import numpy as np

from deep_embeddings import Embedding
from deep_embeddings import LogGaussianPrior, SpikeSlabPrior
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
parser.add_argument("--prior", type=str, default="sslab", choices=["sslab", "gauss", "weibull"], help="Prior to use")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--n_epochs", type=int, default=3000, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--stability_time", type=int, default=100, help="Number of epochs to train before checking stability")
parser.add_argument("--rnd_seed", type=int, default=42, help="Random seed")
parser.add_argument("--gamma", type=float, default=0.5, help="Gamma parameter to balance KL div and reconstruction loss")
parser.add_argument("--params_interval", type=int, default=100, help="Interval to save learned embeddings")
parser.add_argument("--checkpoint_interval", type=int, default=100, help="Interval to save model checkpoints")
parser.add_argument("--mc_samples", type=int, default=5, help="Number of Monte Carlo samples to use for inference")
parser.add_argument("--scale", type=float, default=0.5, help="Scale parameter for log normal prior")
parser.add_argument("--non_zero_weights", type=int, default=5, help="Number of non zero weights for each object")
parser.add_argument("--device_id", type=int, default=0, help="GPU device id")
parser.add_argument("--identifier", type=str, default="", help="Identifier for the experiment")


def train(triplet_path, data_path, model_name, module_name, 
          log_path, modality, fresh=False, load_model=False, tensorboard=False, 
          init_dim=100, prior="sslab", batch_size=256, n_epochs=1000, lr=1e-3, 
          stability_time=500, rnd_seed=42, gamma=0.5, params_interval=100, mc_samples=5,
          checkpoint_interval=500, scale=0.25, non_zero_weights=5, device_id=0, identifier=None):
    
    if fresh and load_model:
        raise ValueError("Cannot load a model and train from scratch at the same time")

    if modality not in ["deep", "behavior"]:
        raise ValueError("Modality must be either 'deep' or 'behavior'")


    torch.manual_seed(rnd_seed)
    np.random.seed(rnd_seed)

    if modality == "behavior":
        n_objects = 1854
    else:
        feature_path = os.path.join(data_path, model_name, module_name, "features.npy")
        features = np.load(feature_path)
        n_objects = features.shape[0]

    
    # model_prior = LogGaussianPrior(n_objects, init_dim, loc=0, scale=scale)
    model_prior = SpikeSlabPrior(n_objects, init_dim)

    
    model = Embedding(model_prior, n_objects, init_dim, non_zero_weights)
    device = torch.device(f"cuda:{device_id}") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)


    # If all data on GPU, num workers need to be 0 and pin memory false
    train_dataset, val_dataset = build_triplet_dataset(triplet_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    n_train = len(train_dataset)
    n_val = len(val_dataset)
    n_mio_samples = str(int((n_train + n_val) / 1e6)) + "mio"
    
    # Print the number of millions in sample size as string
    print("Number of samples in training set: {}".format(n_train / 1e6))
    
    # Join all arguments to create a unique log path
    log_path = os.path.join(log_path, identifier, model_name, n_mio_samples, 
                            modality, prior, str(init_dim), 
                            str(batch_size), str(gamma), str(scale), str(rnd_seed))

    # Build loggers and train the model!
    logger = DeepEmbeddingLogger(log_path, model, fresh, tensorboard, params_interval, checkpoint_interval)
    trainer = EmbeddingTrainer(model, model_prior, train_loader, val_loader, logger, device, 
                               load_model, n_epochs, mc_samples, lr, gamma, stability_time)
    trainer.train()

if __name__ == '__main__':
    args = parser.parse_args()
    train(args.triplet_path, args.data_path, args.model_name, args.module_name, args.log_path, args.modality,
          args.fresh, args.load_model, args.tensorboard, args.init_dim, args.prior, args.batch_size, args.n_epochs, args.lr, args.stability_time,
          args.rnd_seed, args.gamma, args.params_interval, args.mc_samples, args.checkpoint_interval, args.scale, args.non_zero_weights, 
          args.device_id, args.identifier)