#!/usr/bin/env python3
import numpy as np
import torch
import os
import torch.nn.functional as F
import deep_embeddings.utils as utils
from dataclasses import dataclass, field

@dataclass
class TrainingParams:
    lr: float = 0.001
    n_epochs: int = 100
    mc_samples: int = 5 
    stability_time: int = 500

    best_train_loss: float = np.inf
    best_val_loss: float = np.inf
    start_epoch: int = 1
    smallest_dim: int = np.inf
    n_steps_same_dim: int = 0
    
    # store training performance
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    complexity_losses: list = field(default_factory=list)
    log_likelihoods: list = field(default_factory=list)
    dim_over_time: list = field(default_factory=list)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.__dict__:
                continue
            attr = getattr(self, k)
            if isinstance(attr, (int, float)) and isinstance(v, (int, float)):
                setattr(self, k, v)
            elif isinstance(attr, list) and isinstance(v, (int, float)):
                getattr(self, k).append(v)
            else:
                getattr(self, k).extend(v)


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
        lr=0.001,
        stability_time=200,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.logger = logger
        self.params = TrainingParams(lr, n_epochs, mc_samples, stability_time)
        self.log_path = logger.log_path
        self.init_dim = self.model.init_dim
        self._build_optimizer()

        self.prior = prior
        self.device = device
        self.model.to(self.device)
        self.prior.to(self.device)
        self.similarity_mat = similarity_mat.to(device)

    def parse_from_config(self, cfg):
        attrs = ['n_epochs', 'mc_samples',  'lr', 'stability_time']
        for attr in attrs:
            if hasattr(cfg, attr):
                setattr(self.params, attr, getattr(cfg,attr))        

    def _build_optimizer(self):
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.params.lr)

    def update_training_params(self, **kwargs):
        self.params.update(**kwargs)

    def init_model_from_checkpoint(self, checkpoint):
        print("Load model and optimizer from state dict and continue training")
        checkpoint = torch.load(checkpoint)
        self.optim = self.optim.load_state_dict(checkpoint["optim_state_dict"])
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.params = checkpoint["params"]
        self.n_epochs = self.params.start_epoch + self.n_epochs

    
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


    def step_batch(self, indices):
        indices = indices.to(self.device)
        indices = indices.unbind(1)  # convert into two lists of x,y indices

        sim_features = self.similarity_mat[indices]
        embedding, loc, scale = self.model()

        sim_embedding = self.compute_embedding_similarities(embedding, indices)
        log_likelihood = F.mse_loss(sim_features, sim_embedding)

        # log probability of variational distribution
        log_q = utils.normalized_pdf(embedding, loc, scale).log()

        # gaussian prior log probability of the prior distribution
        log_p = self.prior(embedding).log()
        return log_likelihood, log_q, log_p

    def step_dataloader(self, dataloader):
        n_pairwise = len(dataloader.dataset)
        n_batches = len(dataloader)

        complex_losses = torch.zeros(n_batches)
        ll_losses = torch.zeros(n_batches)
        losses = torch.zeros(n_batches)

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
                    (self.params.mc_samples, self.init_dim)
                )
                for s in range(self.params.mc_samples):
                    log_likelihood, log_q, log_p = self.step_batch(indices)
                    complexity_loss = (1 / n_pairwise) * (log_q.sum() - log_p.sum())
                    sampled_likelihoods[s] = log_likelihood.item()

                # validation loss
                loss = torch.mean(sampled_likelihoods).item()

                print(
                    f"Val Batch {k}/{n_batches} Likelihood {loss}",
                    end="\r",
                )

            # def update loss history!
            complex_losses[k] = complexity_loss.item()
            ll_losses[k] = log_likelihood.item()
            losses[k] += loss

        self.params.update(complexity_loss=complex_losses, log_likelihoods=ll_losses)

        if self.model.training:
            self.params.update(train_loss=losses)
        else:
            self.params.update(val_loss=losses)

        mean_loss = sum(losses) / len(losses)
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
        smallest_dim = min(dimensions, self.params.smallest_dim)
        self.params.update(dim_over_time=dimensions, smallest_dim=smallest_dim)
        # self.dimensionality_over_time.append(dimensions)

        # we only check the convergence of the dimensionality after a certain number of epochs
        if epoch < self.params.stability_time: 
            return False

        stability = self.params.dim_over_time[-self.params.stability_time:]
        # if only one dimension is present, the model is stable
        if len(set(stability)) == 1:
            return True
        return False

    def train(self):
        self.model.to(self.device)
        for epoch in range(self.params.start_epoch, self.params.n_epochs + 1):

            train_loss = self.train_one_epoch()
            val_loss = self.evaluate_one_epoch()

            convergence = self.evaluate_convergence(epoch)

            # we log and store intermediate weights concurrently
            log_params = dict(train_loss=train_loss, val_loss=val_loss, model=self.model, epoch=epoch, optim=self.optim,
                              train_loader=self.train_loader, val_loader=self.val_loader, 
                              dim=self.params.smallest_dim, print_prepend='Epoch {}'.format(epoch))

            # TODO add final logging step before it ends!
            if convergence:
                print(f"Stopped training after {epoch} epochs. Model has converged!")
                self.logger.log(log_params)
                self.store_final_embeddings(epoch)
                break
    
            self.logger.log(**log_params)

    def store_final_embeddings(self, epoch):
        pruned_loc, pruned_scale = self.model.sorted_pruned_params()
        f_path = os.path.join(self.log_path, "final_pruned_params.npz")

        with open(f_path, "wb") as f:
            np.savez(f, pruned_loc=pruned_loc, pruned_scale=pruned_scale)