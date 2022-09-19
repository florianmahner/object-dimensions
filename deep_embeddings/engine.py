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
    gamma: float = 0.5  # balances complexity and reconstruction loss
    n_epochs: int = 100
    mc_samples: int = 5
    stability_time: int = 300
    prune_dim: bool = True
    burnin: int = 100

    best_train_loss: float = np.inf
    best_val_loss: float = np.inf
    start_epoch: int = 1
    smallest_dim: int = np.inf
    n_steps_same_dim: int = 0

    # store training performance
    complexity_loss: float = np.inf
    log_likelihoods: float = np.inf
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    dim_over_time: list = field(default_factory=list)
    triplet_choices: list = field(default_factory=list)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.__dict__:
                continue
            attr = getattr(self, k)
            if isinstance(attr, (int, float)) and isinstance(v, (int, float)):
                setattr(self, k, v)
            elif isinstance(attr, list) and isinstance(v, (int, float)):
                getattr(self, k).extend([v])
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
        device="cpu",
        n_epochs=100,
        mc_samples=5,
        lr=0.001,
        gamma=0.5,
        stability_time=200,
        prune_dim=False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.logger = logger
        self.params = TrainingParams(
            lr, gamma, n_epochs, mc_samples, stability_time, prune_dim
        )
        self.log_path = logger.log_path

        self.init_dim = self.model.module.init_dim # NOTE changed this to dataparallel!
        self._build_optimizer()

        self.prior = prior
        self.device = device
        self.model.to(self.device)
        self.prior.to(self.device)


    def parse_from_config(self, cfg):
        attrs = ["n_epochs", "mc_samples", "lr", "gamma", "stability_time", "prune_dim"]
        for attr in attrs:
            if hasattr(cfg, attr):
                setattr(self.params, attr, getattr(cfg, attr))

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
        self.logger = checkpoint["logger"]  # only if exists!


    def step_triplet_batch(self, indices):

        # maybe do this somewhere else to improve speed
        indices = indices.type("torch.LongTensor")
        indices = indices.to(self.device)
        indices = indices.unbind(1)

        ind_i, ind_j, ind_k = indices
        embedding, loc, scale = self.model()

        embedding_i = F.relu(embedding[ind_i])
        embedding_j = F.relu(embedding[ind_j])
        embedding_k = F.relu(embedding[ind_k])

        sim_ij = torch.sum(embedding_i * embedding_j, dim=1)
        sim_ik = torch.sum(embedding_i * embedding_k, dim=1)
        sim_jk = torch.sum(embedding_j * embedding_k, dim=1)

        # compute the log softmax loss, i.e. we just look at the argmax anyways!
        sims = torch.stack([sim_ij, sim_ik, sim_jk])
        log_softmax = F.log_softmax(sims, dim=0)  # i.e. BS x 3

        # compute accuracy for that batch
        # triplet_choices = log_softmax.argmax(0)
        triplet_choices = torch.argmax(log_softmax, 0)
        triplet_accuracy = torch.sum(triplet_choices==0) / len(triplet_choices)

        # these are the most similar, i.e. the argmax, k is the ooo.
        log_softmax_ij = log_softmax[0]

        # compute the cross entropy loss -> we dont need to one hot encode the batch
        # we just use the similarities at index 0, all other targets are 0 and redundant!
        log_likelihood = torch.mean(-log_softmax_ij)
    
        # log probability of variational distribution
        # log_q = utils.log_normal_pdf(embedding, loc, scale)
        log_q = utils.normal_pdf(embedding, loc, scale).log()

        # gaussian prior log probability of the prior distribution
        log_p = self.prior(embedding).log()
        

        return log_likelihood, log_q, log_p, triplet_accuracy


    # NOTE pruned dim is for pairiwise -> need to do this later!
    def step_dataloader(self, dataloader, pruned_dim=None):
        n_pairwise = len(dataloader.dataset)
        n_batches = len(dataloader)
        batch_size = len(dataloader.dataset[0])

        complex_losses = torch.zeros(n_batches)
        ll_losses = torch.zeros(n_batches)
        losses = torch.zeros(n_batches)

        triplet_accuracies = torch.zeros(n_batches)

        for k, indices in enumerate(dataloader):

            beta = 1.0  # NOTE TODO no temp scaling right now!

            # only for trainin batches, not val batches
            if self.model.training:
                # log_likelihood, log_q, log_p = self.step_pairwise_batch(indices)

                log_likelihood, log_q, log_p, accuracy = self.step_triplet_batch(indices)

                # temperature scaling!
                log_likelihood /= beta

                # NOTE short hack to adapt the complexity loss as a function of the batch size!
                complexity_loss = ((batch_size) / n_pairwise) * (log_q.sum() - log_p.sum())
                # complexity_loss = (1 / n_pairwise) * (log_q.sum() - log_p.sum())

                # balance the log likelihood and complexity loss by gamma
                log_likelihood = self.params.gamma * log_likelihood
                complexity_loss = (1 - self.params.gamma) * complexity_loss

                loss = log_likelihood + complexity_loss

                # faster alternative to optim.zero_grad()
                for param in self.model.parameters():
                    param.grad = None

                loss.backward()
                self.optim.step()

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
                    # log_likelihood, log_q, log_p = self.step_batch(indices)
                    log_likelihood, log_q, log_p, accuracy = self.step_triplet_batch(indices)
                    log_likelihood /= beta
                    log_likelihood = self.params.gamma * log_likelihood
                    complexity_loss = (1 / n_pairwise) * (log_q.sum() - log_p.sum())
                    sampled_likelihoods[s] = log_likelihood.item()

                # validation loss
                loss = torch.mean(sampled_likelihoods)

                print(
                    f"Val Batch {k}/{n_batches} Likelihood {loss}",
                    end="\r",
                )

            # def update loss history!
            complex_losses[k] += complexity_loss.item()
            ll_losses[k] += log_likelihood.item()
            losses[k] += loss.item()
            triplet_accuracies[k] += accuracy.item()

        losses = losses.mean().item()

        if self.model.training:
            self.params.update(
                complexity_loss=complex_losses.mean().item(),
                log_likelihoods=ll_losses.mean().item(),
                train_losses=losses,
            )
        else:
            self.params.update(val_loss=losses)

        epoch_accuracy = torch.mean(triplet_accuracies).item()

        return losses, epoch_accuracy

    def train_one_epoch(self):
        self.model.train(True)
        train_loss = self.step_dataloader(self.train_loader)
        return train_loss

    @torch.no_grad()
    def evaluate_one_epoch(self):
        self.model.eval()
        val_loss = self.step_dataloader(self.val_loader)
        return val_loss

    def evaluate_val_loss_convergence(self):
        # NOTE still need to implement this to have an analysis of convergence!
        if self.params.best_val_loss > self.params.val_loss[-1]:
            self.params.update(best_val_loss=self.params.val_loss[-1])

        if (
            self.params.best_val_loss
            not in self.params.val_loss[: -self.params.stability_time]
        ):
            return True

        return False

    def evaluate_convergence(self, epoch):
        # we only check the convergence of the dimensionality after a certain number of epochs
        signal, _, _ = self.model.module.prune_dimensions()
        dimensions = len(signal)
        smallest_dim = min(dimensions, self.params.smallest_dim)
        self.params.update(dim_over_time=dimensions, smallest_dim=smallest_dim)

        stability = self.params.dim_over_time[-self.params.stability_time :]

        if epoch < self.params.stability_time:
            return False

        # if only one dimension is present, the model is stable
        if len(set(stability)) == 1:
            return True

        return False

    def train(self):
        self.model.to(self.device)
        for epoch in range(self.params.start_epoch, self.params.n_epochs + 1):

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.evaluate_one_epoch()

            # evaluate the convergence first!
            convergence = self.evaluate_convergence(epoch)
            # convergence = self.evaluate_val_loss_convergence()

            # we log and store intermediate weights concurrently
            log_params = dict(
                train_loss=train_loss,
                train_acc=train_acc, 
                train_ll=self.params.log_likelihoods,
                train_complexity=self.params.complexity_loss,
                val_loss=val_loss,
                val_acc=val_acc,
                model=self.model,
                epoch=epoch,
                logger=self.logger,
                optim=self.optim,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                dim=self.params.smallest_dim,
                print_prepend="Epoch {}".format(epoch),
            )

            # TODO add final logging step before it ends!
            if convergence:
                print(f"Stopped training after {epoch} epochs. Model has converged!")
                self.logger.log(**log_params)
                self.store_final_embeddings(epoch)
                break

            self.logger.log(**log_params)

    def store_final_embeddings(self, epoch):
        params = self.model.module.sorted_pruned_params()
        f_path = os.path.join(self.log_path, "final_pruned_params.npz")

        with open(f_path, "wb") as f:
            np.savez(f, **params)
