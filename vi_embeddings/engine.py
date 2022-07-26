#!/usr/bin/env python3
import numpy as np
import torch
import os
from copy import deepcopy
import logging
import torch.nn.functional as F
import vi_embeddings.utils as utils
from dataclasses import dataclass


@dataclass
class RunConfig:
    n_epochs: int  # maximal number of epochs to train the model
    mc_samples: int  # number of samples to take from the posterior
    checkpoint_interval: int  # how often to save a checkpoint
    params_interval: int  # how often to save the weights
    stability_interval: int  # interval to oberserve change in ll loss
    best_train_loss: float = np.inf
    best_val_loss: float = np.inf
    start_epoch: int = 1
    # dim_over_time: list = None
    smallest_dim: int = np.inf
    n_steps_same_dim: int = 0


class MLTrainer:
    "Trainer class to run the model"

    def __init__(
        self,
        model,
        prior,
        train_loader,
        val_loader,
        logger,  # defines the log path!
        similarity_mat,
        device="cpu",
        n_epochs=100,
        mc_samples=5,
        checkpoint_interval=50,
        lr=0.001,
        stability_time=200,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.similarity_mat = similarity_mat.to(device)
        self.logger = logger

        self.lr = lr
        self.mc_samples = mc_samples
        self.n_epochs = n_epochs
        self.log_path = logger.log_path
        self.checkpoint_interval = checkpoint_interval
        self.stability_time = stability_time
        self._init_training()

        self.prior = prior
        self.device = device
        self.model.to(self.device)
        self.prior.to(self.device)

    def update_training_params(self):
        pass

    def parse_from_config(self, cfg):
        attrs = ['n_epochs', 'mc_samples', 
                 'checkpoint_interval', 'lr', 
                 'stability_time']
        for attr in attrs:
            if hasattr(cfg, attr):
                setattr(self, attr, getattr(cfg,attr))

    def _init_training(self):
        self.start_epoch = 1
        self.loss = 0
        self.best_loss = np.inf
        self.smallest_dim = np.inf
        self.n_steps_same_dim = 0
        self.dimensionality_over_time = []
        self.init_dimensions = self.model.init_dim
        self._build_optimizer()

    def _build_optimizer(self):
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _init_optimizer(self, state_dict):
        self.optim = self.optim.load_state_dict(state_dict)

    def init_model_from_checkpoint(self, checkpoint):
        print("Load model and optimizer from state dict and continue training")
        checkpoint = torch.load(checkpoint)
        saved_args = checkpoint["args"]

        self._init_optimizer(checkpoint["optim_state_dict"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.start_epoch = checkpoint["epoch"]
        self.loss = checkpoint["loss"]
        self.n_epochs = self.start_epoch + self.n_epochs
        self.lr = saved_args.lr

    def save_checkpoint(self, epoch):
        # save model!
        m_path = os.path.join(self.log_path, "checkpoint_epoch_{}.tar")
        torch.save(
            {
                "model_state_dict": deepcopy(self.model.state_dict()),
                "optim_state_dict": deepcopy(self.optim.state_dict()),
                "train_loader": self.train_loader,
                "val_loader": self.val_loader,
                "epoch": epoch,
            },
            m_path.format(epoch),
        )

    def step_batch(self, indices):
        indices = indices.to(self.device)
        indices = indices.unbind(1)  # convert into two lists of x,y indices

        sim_features = self.similarity_mat[indices]
        # sim_features = sim_features.to(self.device)

        embedding, loc, scale = self.model()

        sim_embedding = self.compute_embedding_similarities(embedding, indices)
        log_likelihood = F.mse_loss(sim_features, sim_embedding)

        # log probability of variational distribution
        log_q = utils.normalized_pdf(embedding, loc, scale).log()

        # gaussian prior log probability of the prior distribution
        log_p = self.prior(embedding).log()
        return log_likelihood, log_q, log_p

    def step_dataloader(self, dataloader):
        loss_history = []
        n_pairwise = len(dataloader.dataset)
        n_batches = len(dataloader)

        for k, indices in enumerate(dataloader):

            # only for trainin batches, not val batches
            if self.model.training:
                log_likelihood, log_q, log_p = self.step_batch(indices)
                complexity_loss = (1 / n_pairwise) * (log_q.sum() - log_p.sum())

                loss = log_likelihood + complexity_loss

                # faster alternative to optim.zero_grad()
                for param in self.model.parameters():
                    param.grad = None

                loss.backward()
                self.optim.step()
                loss = loss.item()

                print(
                    f"Train Batch {k}/{n_batches} Likelihood {log_likelihood.item()}, Complexity {complexity_loss.item()}",
                    end="\r",
                )

            # we do mc sampling of the variational posterior for validation batches
            else:
                sampled_likelihoods = torch.zeros(
                    (self.mc_samples, self.init_dimensions)
                )
                for s in range(self.mc_samples):
                    log_likelihood, _, _ = self.step_batch(indices)
                    sampled_likelihoods[s] = log_likelihood.item()

                # validation loss
                loss = torch.mean(sampled_likelihoods).item()

                print(
                    f"Val Batch {k}/{n_batches} Likelihood {loss}",
                    end="\r",
                )

            loss_history.append(loss)

        mean_loss = sum(loss_history) / len(loss_history)
        return mean_loss

    def train_one_epoch(self):
        self.model.train(True)
        train_loss = self.step_dataloader(self.train_loader)
        return train_loss

    @torch.no_grad()
    def evaluate_one_epoch(self):
        self.model.eval()
        val_loss = self.step_dataloader(self.val_loader)
        return val_loss

    def evaluate_convergence(self, epoch):
        signal, _, _ = self.model.prune_dimensions()
        dimensions = len(signal)
        self.dimensionality_over_time.append(dimensions)

        # we only check the convergence of the dimensionality after a certain number of epochs
        if epoch < self.stability_time: 
            return False

        stability = self.dimensionality_over_time[-self.stability_time:]
        # if only one dimension is present, the model is stable
        if len(set(stability)) == 1:
            return True
        return False

    def train(self):
        self.model.to(self.device)
        for epoch in range(self.start_epoch, self.n_epochs + 1):

            train_loss = self.train_one_epoch()
            val_loss = self.evaluate_one_epoch()

            if self.evaluate_convergence(epoch):
                print(f"Stopped training after {epoch} epochs. Model has converged!")
                self.save_checkpoint(epoch)
                self.store_final_embeddings(epoch)
                break

            # we log and store intermediate weights concurrently
            log_params = dict(train_loss=train_loss, val_loss=val_loss)
            self.logger.log(**log_params, prepend='Epoch {}'.format(epoch))
    
            if epoch % self.checkpoint_interval == 0:
                self.save_checkpoint(epoch)

    def store_final_embeddings(self, epoch):
        pruned_loc, pruned_scale = self.model.sorted_pruned_params()
        f_path = os.path.join(self.log_path, "final_pruned_params.npz")

        with open(f_path, "wb") as f:
            np.savez(f, pruned_loc=pruned_loc, pruned_scale=pruned_scale)

    def compute_embedding_similarities(self, embedding, indices):
        indices_i, indices_j = indices

        # NOTE we dont activate right now!
        # embeddings_i = F.relu(embedding[indices_i])
        # embeddings_j = F.relu(embedding[indices_j])

        embeddings_i = embedding[indices_i]
        embeddings_j = embedding[indices_j]

        sim_embedding = torch.sum(
            embeddings_i * embeddings_j, dim=1
        )  # this is the dot produt of the embedding vectors
        return sim_embedding
