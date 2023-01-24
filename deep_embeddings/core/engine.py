#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os
import glob

import numpy as np
import torch.nn.functional as F


class Params(object):
    r"""The class stores the training configuration of the training pipeline
    and updates the results depending on the type of the keyword arguments in the update (e.g. list, array)"""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        for key in (
            "train_kl",
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

    def __getitem__(self, key):
        return self.__dict__[key]

    def update(self, **kwargs):
        """Update the parameters of the model depending on type.
        If the type is a list, append the value to the list, else set new value as attribute"""
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

    def save(self, path):
        """Save the parameters of the model as a dictionary"""
        np.savez(path, **self.__dict__)


class EmbeddingTrainer(object):
    """Trainer class that runs the entire optimzation of learing the embedding, storing and saving checkpoints etc."""

    def __init__(
        self,
        model,
        prior,
        train_loader,
        val_loader,
        logger,
        device="cpu",
        load_model=False,
        n_epochs=100,
        mc_samples=5,
        lr=0.001,
        beta=1.0,
        stability_time=200,
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
        )

        self.log_path = logger.log_path
        self._build_optimizer()
        self.prior = prior
        self.device = device

    def parse_from_args(self, args):
        attrs = ["n_epochs", "mc_samples", "lr", "beta", "stability_time", "load_model"]
        for attr in attrs:
            if hasattr(args, attr):
                setattr(self.params, attr, getattr(args, attr))

    def _build_optimizer(self):
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.params.lr)

    def init_model_from_checkpoint(self):
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
            self.params = params
            self.params.n_epochs = checkpoint["epoch"] + self.params.n_epochs
            self.params.start_epoch = checkpoint["epoch"] + 1
            self.logger = checkpoint["logger"]  # only if exists!

    def calculate_likelihood(self, embedding, indices):
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
        triplet_accuracy = torch.sum(triplet_choices == 0) / len(triplet_choices)

        return nll, triplet_accuracy

    def _kl_divergence(self, mu_prior, mu_q, scale_p, scale_q):
        """Compute the KL divergence between two Log Gaussians. This is the same as the KL divergence
        between two Gaussians. see
        https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians"""
        kl = (
            torch.log(scale_p / scale_q)
            + (scale_q**2 + (mu_q - mu_prior) ** 2) / (2 * scale_p**2)
            - 0.5
        )

        return kl

    def calculate_complexity(self, embedding, loc, scale):
        """Compute the KL divergence between the prior and the variational posterior"""
        n_train = len(self.train_loader.dataset)
        log_q = self.prior.log_pdf(embedding, loc, scale)
        log_p = self.prior(embedding)
        kl_div = log_q.sum() - log_p.sum()
        kl_div = kl_div / n_train

        return kl_div

    def step_triplet_batch(self, indices):
        """Step the model for a single batch of data and extract embedding triplets"""
        embedding, loc, scale = self.model()
        nll, triplet_accuracy = self.calculate_likelihood(embedding, indices)
        kl_div = self.calculate_complexity(embedding, loc, scale)

        return nll, kl_div, triplet_accuracy

    def step_dataloader(self, dataloader):
        """Step the model for a single epoch"""
        n_batches = len(dataloader)
        kl_losses = torch.zeros(n_batches, device=self.device)
        nll_losses = torch.zeros(n_batches, device=self.device)
        triplet_accuracies = torch.zeros(n_batches, device=self.device)

        for k, indices in enumerate(dataloader):

            if self.model.training:
                nll, kl_div, accuracy = self.step_triplet_batch(indices)
                loss = nll + self.params.beta * kl_div

                # faster alternative to optim.zero_grad()
                for param in self.model.parameters():
                    param.grad = None

                loss.backward()
                self.optim.step()

                print(
                    f"Train Batch {k}/{n_batches}",
                    end="\r",
                )

            # we do mc sampling of the variational posterior for validation batches
            else:
                sampled_likelihoods = torch.zeros(self.params.mc_samples)
                for s in range(self.params.mc_samples):
                    embedding = self.model()[0]
                    nll, accuracy = self.calculate_likelihood(embedding, indices)
                    sampled_likelihoods[s] = nll.detach()

                nll = torch.mean(sampled_likelihoods)

                print(
                    f"Val Batch {k}/{n_batches}",
                    end="\r",
                )

            nll_losses[k] = nll.detach()
            triplet_accuracies[k] = accuracy.detach()

            if self.model.training:
                kl_losses[k] = kl_div.detach()

        if self.model.training:
            epoch_loss = nll_losses.mean().item() + kl_losses.mean().item()

            self.params.update(
                train_kl=kl_losses.mean().item(),
                train_nll=nll_losses.mean().item(),
            )
        else:
            epoch_loss = nll_losses.mean().item()
            self.params.update(
                val_nll=epoch_loss,
            )

        epoch_accuracy = torch.mean(triplet_accuracies).item()

        return epoch_loss, epoch_accuracy

    def train_one_epoch(self):
        self.model.train(True)
        train_loss = self.step_dataloader(self.train_loader)

        return train_loss

    @torch.no_grad()
    def evaluate_one_epoch(self):
        self.model.eval()
        val_loss = self.step_dataloader(self.val_loader)
        return val_loss

    def evaluate_convergence(self):
        """We evaluate convergence as the representational stability of the number of dimensions across a certain time
        frame of epochs"""
        signal, _, _ = self.model.prune_dimensions()
        dimensions = len(signal)
        self.params.update(dim_over_time=dimensions)

        stability = self.params.dim_over_time[-self.params.stability_time :]

        if self.epoch < self.params.stability_time:
            return False

        # if only one dimension is present, the model is stable
        if len(set(stability)) == 1:
            return True

        return False

    def train(self):
        if self.params.load_model:
            self.init_model_from_checkpoint()

        self.model.to(self.device)
        self.prior.to(self.device)
        self.batch_size = self.train_loader.batch_size

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
                    train_kl=self.params.train_kl,
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
                    log_params[
                        "final"
                    ] = True  # we also log all things that dont have an update interval
                    self.logger.log(**log_params)
                    self.store_final_embeddings()
                    break

                self.logger.log(**log_params)

        except KeyboardInterrupt:
            print("Training interrupted by user. Saving model and exiting.")
            self.store_final_embeddings()

    def store_final_embeddings(self):
        params = self.model.sorted_pruned_params()
        f_path = os.path.join(self.log_path, "params", "parameters.npz")
        try:
            os.makedirs(os.path.dirname(f_path), exist_ok=True)
        except OSError as e:
            print(
                "Could not create directory for storing parameters, since {}".format(e)
            )

        self.params.update(
            pruned_q_mu=params["pruned_q_mu"],
            pruned_q_var=params["pruned_q_var"],
            embedding=params["embedding"],
        )
        self.params.save(f_path)
