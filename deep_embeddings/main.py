#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import toml
import os
import random
import argparse

import numpy as np

from deep_embeddings.core.engine import EmbeddingTrainer
from deep_embeddings.core.model import VariationalEmbedding, DeterministicEmbedding
from deep_embeddings.core.priors import SpikeSlabPrior, LogGaussianPrior

from deep_embeddings import ExperimentParser, build_triplet_dataset
from deep_embeddings import DeepEmbeddingLogger


parser = ExperimentParser(description="Main training script for deep embeddings")
parser.add_argument(
    "--method", 
    type=str,
    default="variational",
    choices=("variational", "deterministic"),
    help="Type of embedding to train. Variational = VICE, deterministic = SPoSE"
)
parser.add_argument("--triplet_path", type=str, help="Path to the triplet file")
parser.add_argument(
    "--log_path",
    type=str,
    default="./results",
    help="Path to store all training outputs and model checkpoints",
)
parser.add_argument(
    "--modality",
    type=str,
    default="deep",
    choices=("deep", "behavior"),
    help="Modality to train on",
)
parser.add_argument(
    "--fresh",
    default=False,
    action="store_true",
    help="""Start clean and delete old path content. Otherwise
continue training from checkpoint if it exists and load the previous config file from that directory""",
)
parser.add_argument(
    "--load_model",
    default=False,
    action="store_true",
    help="Load a pretrained model from log path",
)
parser.add_argument(
    "--tensorboard",
    default=False,
    action="store_true",
    help="Use tensorboard to log training",
)
parser.add_argument(
    "--init_dim",
    type=int,
    default=100,
    help="Initial dimensionality of the latent space",
)
parser.add_argument(
    "--prior",
    type=str,
    default="sslab",
    choices=["sslab", "gauss"],
    help="Select the prior to use for the variational embedding",
)
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument(
    "--n_epochs", type=int, default=3000, help="Number of epochs to train for"
)
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument(
    "--stability_time",
    type=int,
    default=100,
    help="Number of epochs to train before checking stability",
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument(
    "--beta",
    type=float,
    default=1.0,
    help="Beta parameter to balance KL div and reconstruction loss",
)
parser.add_argument(
    "--params_interval",
    type=int,
    default=100,
    help="Interval to save learned embeddings",
)
parser.add_argument(
    "--checkpoint_interval",
    type=int,
    default=100,
    help="Interval to save model checkpoints",
)
parser.add_argument(
    "--mc_samples",
    type=int,
    default=5,
    help="Number of Monte Carlo samples to use for inference",
)
parser.add_argument(
    "--scale", type=float, default=0.5, help="Scale parameter for log normal prior"
)
parser.add_argument(
    "--non_zero_weights",
    type=int,
    default=5,
    help="Number of non zero weights for each object",
)
parser.add_argument("--device_id", type=int, default=0, help="GPU device id")
parser.add_argument(
    "--identifier", type=str, default="", help="Identifier for the experiment"
)


def _parse_number_of_objects(triplet_path, modality):
    if modality == "behavior":
        return 1854
    else:
        base_path = os.path.dirname(triplet_path)
        feature_path = os.path.join(base_path, "features.npy")
        features = np.load(feature_path)
        return features.shape[0]


def _set_global_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _build_prior(prior, n_objects, init_dim, scale):
    if prior == "sslab":
        return SpikeSlabPrior(n_objects, init_dim)
    elif prior == "gauss":
        return LogGaussianPrior(n_objects, init_dim, loc=0.0, scale=scale)
    else:
        raise ValueError("Unknown prior: {}".format(prior))


def _convert_samples_to_string(train_dataset, val_dataset):
    """Finds the combined number of training and validation triplets
    and converts them to a printable representation in number of millions"""
    n_samples = len(train_dataset) + len(val_dataset)

    n_samples = round(n_samples / 1e6, 2)
    n_samples = str(n_samples)
    n_samples = n_samples.rstrip("0")
    n_samples = n_samples + "mio"
    return n_samples


def load_args(args, log_path, fresh):
    load_model = args.load_model
    # If we continue training we load the previous parameters
    if os.path.exists(os.path.join(log_path, "config.toml")) and not fresh:
        print(
            "Loading previous config file from the directory {} to continue training".format(
                log_path
            )
        )
        toml_config = toml.load(os.path.join(log_path, "config.toml"))
        args = argparse.Namespace(**toml_config)
    else:
        with open(os.path.join(log_path, "config.toml"), "w") as f:
            toml.dump(vars(args), f)

    args.load_model = load_model
    return args


def _build_model(args):
    n_objects = _parse_number_of_objects(args.triplet_path, args.modality)
    if args.method == "variational":
        model_prior = _build_prior(args.prior, n_objects, args.init_dim, args.scale)
        model = VariationalEmbedding(model_prior, n_objects, args.init_dim, args.non_zero_weights)
        return model, model_prior
    else:
        model = DeterministicEmbedding(n_objects, args.init_dim)
        return model, None

def _check_args(args):
    if args.fresh and args.load_model:
        raise ValueError("Cannot load a model and train from scratch at the same time")

def train(args):
    device = (
        torch.device(f"cuda:{args.device_id}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dataset, val_dataset = build_triplet_dataset(args.triplet_path, device=device, modality=args.modality)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
        generator=g,
        num_workers=16,
        worker_init_fn=_set_global_seed,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=16,
        generator=g,
        worker_init_fn=_set_global_seed
    )
    n_samples = _convert_samples_to_string(train_dataset, val_dataset)

    # If we are training a deterministic embedding using MLE this is the same as a variational point estimate with a uniform prior
    if args.method == "deterministic":
        args.prior = "uniform"

    # Build the logpath
    if args.modality == "behavior":
        log_path = os.path.join(
            args.log_path,
            args.identifier,
            args.modality,
            n_samples,
            args.prior,
            str(args.init_dim),
            str(args.batch_size),
            str(args.beta),
            str(args.seed),
        )
    else:
        model_name, module_name = args.triplet_path.split("/")[-3:-1]
        log_path = os.path.join(
            args.log_path,
            args.identifier,
            args.modality,
            model_name,
            module_name,
            n_samples,
            args.prior,
            str(args.init_dim),
            str(args.batch_size),
            str(args.beta),
            str(args.seed),
        )

    # Save all configuration parameters in the directory
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    args = load_args(args, log_path, args.fresh)
    _set_global_seed(args.seed)    
    model, model_prior = _build_model(args)
    model.to(device)

    # Build loggers and train the model!
    logger = DeepEmbeddingLogger(
        log_path,
        model,
        args.fresh,
        args.tensorboard,
        args.params_interval,
        args.checkpoint_interval,
    )
    trainer = EmbeddingTrainer(
        model,
        model_prior,
        train_loader,
        val_loader,
        logger,
        device,
        args.load_model,
        args.n_epochs,
        args.mc_samples,
        args.lr,
        args.beta,
        args.stability_time,
        args.method
    )
    trainer.train()

if __name__ == "__main__":
    args = parser.parse_args()
    _check_args(args)
    train(args)
