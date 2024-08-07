#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os
import glob
import argparse

import numpy as np
import torch.nn.functional as F

from typing import List, Dict, Union, Any, Tuple
from torch.utils.data import DataLoader

from .embedding import (
    VariationalEmbedding,
    DeterministicEmbedding,
)
from .priors import SpikeSlabPrior, LogGaussianPrior
from .logging import ExperimentLogger


class Params(object):
    r"""The class stores the training configuration of the training pipeline
    and updates the results depending on the type of the keyword arguments in the update (e.g. list, array)
    """

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        self.__dict__.update(kwargs)
        for key in (
            "train_complexity",
            "train_nll",
            "train_loss",
            "val_loss",
            "train_acc",
            "val_nll",
            "val_acc",
            "dim_over_time",
        ):
            setattr(self, key, [])
        self.start_epoch = 1

    def __getitem__(self, key: str) -> Union[int, float, List, np.ndarray]:
        return self.__dict__[key]

    def update(self, **kwargs: Dict[str, Union[int, float, List]]) -> None:
        """Update the parameters of the model depending on type.
        If the type is a list, append the value to the list, else set new value as attribute
        """
        for k, v in kwargs.items():
            if k not in self.__dict__:
                setattr(self, k, v)
            attr = getattr(self, k)
            if isinstance(attr, (int, float)) and isinstance(v, (int, float)):
                setattr(self, k, v)
            elif isinstance(attr, list) and isinstance(v, (int, float)):
                getattr(self, k).extend([v])
            elif isinstance(attr, list) and isinstance(v, list):
                getattr(self, k).extend(v)
            else:
                setattr(self, k, v)

    def save(self, path: str) -> None:
        """Save the parameters of the model as a dictionary"""
        np.savez_compressed(path, **self.__dict__)


class EmbeddingTrainer(object):
    """Trainer class that runs the entire optimzation of learing the embedding, storing and saving checkpoints etc."""

    def __init__(
        self,
        model: Union[VariationalEmbedding, DeterministicEmbedding],
        prior: Union[SpikeSlabPrior, LogGaussianPrior],
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: ExperimentLogger,
        device: str = "cpu",
        load_model: bool = False,
        n_epochs: int = 100,
        mc_samples: int = 5,
        lr: float = 0.001,
        beta: float = 1.0,
        stability_time: int = 200,
        method: str = "variational",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

        self.params = Params(
            lr=lr,
            init_dim=model.init_dim,
            beta=beta,
            n_epochs=n_epochs,
            mc_samples=mc_samples,
            stability_time=stability_time,
            load_model=load_model,
            method=method,
        )

        self.log_path = logger.log_path
        self._build_optimizer()
        self.prior = prior
        self.device = device
        self.method = method

    def parse_from_args(self, args: argparse.Namespace):
        attrs = ["n_epochs", "mc_samples", "lr", "beta", "stability_time", "load_model"]
        for attr in attrs:
            if hasattr(args, attr):
                setattr(self.params, attr, getattr(args, attr))

    def _build_optimizer(self) -> None:
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.params.lr)

    def init_model_from_checkpoint(self) -> None:
        print("Load model and optimizer from state dict to resume training")

        # Find file with .tar ending in directory
        checkpoint_path = os.path.join(self.log_path, "checkpoints/*.tar")
        checkpoints = glob.glob(checkpoint_path)

        if not checkpoints:
            print(
                "No checkpoint found in {}. Cannot resume training, start fresh instead".format(
                    self.log_path
                )
            )
        else:
            checkpoint = torch.load(checkpoints[0])
            self.optim.load_state_dict(checkpoint["optim_state_dict"])
            self.model.load_state_dict(checkpoint["model_state_dict"])
            params = checkpoint["params"]
            params.stability_time = self.params.stability_time
            params.n_epochs = self.params.n_epochs
            self.params = params
            self.params.start_epoch = checkpoint["epoch"] + 1
            self.logger = checkpoint["logger"]  # only if exists!

    def calculate_likelihood(
        self, embedding: torch.Tensor, indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the negative log likelihood of the data given the embedding"""
        indices = indices.unbind(1)
        ind_i, ind_j, ind_k = indices
        embedding_i = F.relu(embedding[ind_i])
        embedding_j = F.relu(embedding[ind_j])
        embedding_k = F.relu(embedding[ind_k])

        sim_ij = torch.einsum("ij,ij->i", embedding_i, embedding_j)
        sim_ik = torch.einsum("ij,ij->i", embedding_i, embedding_k)
        sim_jk = torch.einsum("ij,ij->i", embedding_j, embedding_k)

        # Compute the log softmax loss, i.e. we just look at the argmax anyways!
        sims = torch.cat([sim_ij, sim_ik, sim_jk]).view(
            3, -1
        )  # faster than torch.stack

        log_softmax = F.log_softmax(sims, dim=0)  # i.e. 3 x batch_size

        # Theis the most similar (sim_ij), i.e. the argmax, k is the ooo.
        log_softmax_ij = log_softmax[0]

        # Compute the cross entropy loss -> we dont need to one hot encode the batch
        # We just use the similarities at index 0, all other targets are 0 and redundant!
        nll = torch.mean(-log_softmax_ij)

        # Compute accuracy for that batch
        triplet_choices = torch.argmax(log_softmax, 0)
        triplet_accuracy = (triplet_choices == 0).float().mean()

        return nll, triplet_accuracy

    def _kl_divergence(
        self,
        mu_prior: torch.Tensor,
        mu_q: torch.Tensor,
        scale_p: torch.Tensor,
        scale_q: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the KL divergence between two Log Gaussians. This is the same as the KL divergence
        between two Gaussians. see
        https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        """
        kl = (
            torch.log(scale_p / scale_q)
            + (scale_q**2 + (mu_q - mu_prior) ** 2) / (2 * scale_p**2)
            - 0.5
        )

        return kl

    def calculate_kl_div(
        self, embedding: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Compute the KL divergence between the prior and the variational posterior"""
        n_train = len(self.train_loader.dataset)
        log_q = self.prior.log_pdf(embedding, loc, scale)
        log_p = self.prior(embedding)
        kl_div = log_q.sum() - log_p.sum()
        kl_div = kl_div / n_train
        return kl_div

    def calculate_spose_complexity(self, embedding: torch.Tensor) -> torch.Tensor:
        """Compute the complexity loss for the SPOSE model"""
        l1_pen = self.model.l1_regularization()
        n_items = self.model.n_objects
        pos_pen = torch.sum(
            F.relu(-embedding)
        )  # positivity constraint to enforce non-negative values in embedding matrix
        complexity_loss = (self.params.beta / n_items) * l1_pen
        complexity_loss = 0.01 * pos_pen + complexity_loss

        return complexity_loss

    def get_nitems(self) -> int:
        train_triplets = self.train_loader.dataset.triplet_indices
        # number of unique items in the data matrix
        n_items = torch.max(train_triplets).item()
        if torch.min(train_triplets).item() == 0:
            n_items += 1
        return n_items

    def step_triplet_batch(
        self, indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step the model for a single batch of data and extract embedding triplets"""
        if self.method == "variational":
            embedding, loc, scale = self.model()
            complex_loss = self.calculate_kl_div(embedding, loc, scale)
        else:
            embedding = self.model()
            complex_loss = self.calculate_spose_complexity(embedding)

        nll, triplet_accuracy = self.calculate_likelihood(embedding, indices)

        return nll, complex_loss, triplet_accuracy

    def variational_evaluation(
        self, indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from the variational posterior and compute the likelihood of the data"""
        sampled_likelihoods = torch.zeros(self.params.mc_samples)
        sampled_accuracies = torch.zeros(self.params.mc_samples)
        for s in range(self.params.mc_samples):
            embedding = self.model()[0]
            nll, accuracy = self.calculate_likelihood(embedding, indices)
            sampled_likelihoods[s] = nll.detach()
            sampled_accuracies[s] = accuracy.detach()

        nll = torch.mean(sampled_likelihoods)
        accuracy = torch.mean(sampled_accuracies)

        return nll, accuracy

    def step_dataloader(
        self, dataloader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Step the model for a single epoch"""
        n_batches = len(dataloader)
        complex_losses = torch.zeros(n_batches, device=self.device)
        nll_losses = torch.zeros(n_batches, device=self.device)
        triplet_accuracies = torch.zeros(n_batches, device=self.device)

        for k, indices in enumerate(dataloader):
            if self.model.training:
                nll, complex_loss, accuracy = self.step_triplet_batch(indices)
                complex_losses[k] = complex_loss.detach()

                if self.method == "variational":
                    loss = nll + self.params.beta * complex_loss
                else:
                    loss = nll + complex_loss

                # faster alternative to optim.zero_grad()
                for param in self.model.parameters():
                    param.grad = None

                loss.backward()
                self.optim.step()

                print(
                    f"Train Batch {k}/{n_batches}",
                    end="\r",
                )

            # Do a variational evaluation if we are not training
            if self.model.training == False:
                if self.method == "variational":
                    nll, accuracy = self.variational_evaluation(indices)
                else:
                    embedding = self.model()
                    nll, accuracy = self.calculate_likelihood(embedding, indices)

                    print(
                        f"Val Batch {k}/{n_batches}",
                        end="\r",
                    )

            nll_losses[k] = nll.detach()
            triplet_accuracies[k] = accuracy.detach()

        if self.model.training:
            epoch_loss = nll_losses.mean().item() + complex_losses.mean().item()

            self.params.update(
                train_complexity=complex_losses.mean().item(),
                train_nll=nll_losses.mean().item(),
            )
        else:
            epoch_loss = nll_losses.mean().item()
            self.params.update(
                val_nll=epoch_loss,
            )

        epoch_accuracy = torch.mean(triplet_accuracies).item()

        return epoch_loss, epoch_accuracy

    def train_one_epoch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.train(True)
        train_loss, train_acc = self.step_dataloader(self.train_loader)
        return train_loss, train_acc

    @torch.no_grad()
    def evaluate_one_epoch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        val_loss, val_acc = self.step_dataloader(self.val_loader)
        return val_loss, val_acc

    def evaluate_convergence(self) -> bool:
        """We evaluate convergence as the representational stability of the number of dimensions across a certain time
        frame of epochs"""
        signal = self.model.prune_dimensions()[0]
        dimensions = len(signal)

        self.params.update(dim_over_time=dimensions)

        stability = self.params.dim_over_time[-self.params.stability_time :]

        if self.epoch < self.params.stability_time:
            return False

        # if only one dimension is present, the model is stable
        if len(set(stability)) == 1:
            return True

        return False

    def train(self) -> None:
        if self.params.load_model:
            self.init_model_from_checkpoint()

        self.model.to(self.device)
        if self.prior:
            self.prior.to(self.device)
        self.batch_size = self.train_loader.batch_size
        print("Start training the model")

        try:
            for self.epoch in range(self.params.start_epoch, self.params.n_epochs + 1):
                # Train loss is combined log likelihood and complexity. The val loss is only the log likelihood
                # average over multiple monte carlo samples!
                train_loss, train_acc = self.train_one_epoch()
                val_loss, val_acc = self.evaluate_one_epoch()
                convergence = self.evaluate_convergence()

                # Update our training params
                self.params.update(
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_acc=train_acc,
                    val_acc=val_acc,
                    epoch=self.epoch,
                )

                # Update our log params
                log_params = dict(
                    train_loss=train_loss,
                    train_acc=train_acc,
                    train_nll=self.params.train_nll,
                    train_complexity=self.params.train_complexity,
                    val_loss=val_loss,
                    val_nll=self.params.val_nll,
                    val_acc=val_acc,
                    model=self.model,
                    beta=self.params.beta,
                    epoch=self.epoch,
                    logger=self.logger,
                    optim=self.optim,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    dim=self.params.dim_over_time[-1],
                    params=self.params,
                    print_prepend="Epoch {}".format(self.epoch),
                )

                if convergence or (self.epoch == self.params.n_epochs):
                    print(
                        f"Stopped training after {self.epoch} epochs. Model has converged or max number of epochs have been reached!"
                    )
                    log_params["final"] = (
                        True  # we also log all things that dont have an update interval
                    )
                    self.logger.log(**log_params)
                    self.store_final_embeddings()
                    break

                self.logger.log(**log_params)

        except KeyboardInterrupt:
            print("Training interrupted by user. Saving model and exiting.")
            self.store_final_embeddings()

    def store_final_embeddings(self) -> None:
        pruned_params = self.model.sorted_pruned_params()
        params = self.model.detached_params()
        f_path = os.path.join(self.log_path, "params", "parameters.npz")
        try:
            os.makedirs(os.path.dirname(f_path), exist_ok=True)
        except OSError as e:
            print(
                "Could not create directory for storing parameters, since {}".format(e)
            )

        self.params.update(**pruned_params, **params)
        self.params.save(f_path)
