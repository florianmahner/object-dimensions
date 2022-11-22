#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os
import glob

import numpy as np
import torch.nn.functional as F
from deep_embeddings.utils import utils 


class Params(object):
    """ The class stores the training configuration of the training pipeline 
    and updates the results depending on the type of the keyword arguments in the update (e.g. list, array)"""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        for key in ("train_complexity", "train_nll", "train_loss", "val_loss", "train_acc", "val_nll", "val_complexity", "val_acc", "dim_over_time"):
            setattr(self, key, [])
        self.start_epoch = 1
        
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def update(self, **kwargs):
        """ Update the parameters of the model depending on type. 
        If the type is a list, append the value to the list, else set new value as attribute """
        for k, v in kwargs.items():
            if k not in self.__dict__:
                continue
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
        """ Save the parameters of the model as a dictionary """
        np.savez(path, **self.__dict__)


class EmbeddingTrainer(object):
    """ Trainer class that runs the entire optimzation of learing the embedding, storing and saving checkpoints etc. """
    def __init__(
        self,
        model,
        prior,
        train_loader,
        val_loader,
        logger,  # defines the log path!
        device="cpu",
        load_model=False,
        n_epochs=100,
        mc_samples=5,
        lr=0.001,
        gamma=0.5,
        stability_time=200, 
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

        self.params = Params(lr=lr, 
                             init_dim=model.init_dim,
                             gamma=gamma, 
                             n_epochs=n_epochs, 
                             mc_samples=mc_samples, 
                             stability_time=stability_time, 
                             load_model=load_model)

        self.log_path = logger.log_path
        self._build_optimizer()
        self.prior = prior
        self.device = device
        self.model.to(self.device)

    def parse_from_args(self, args):
        attrs = ["n_epochs", "mc_samples", "lr", "gamma", "stability_time", "load_model"]
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
            print("No checkpoint found in {}. Cannot resume training, start fresh instead".format(self.log_path))
        else:
            checkpoint = torch.load(checkpoints[0])
            self.optim.load_state_dict(checkpoint["optim_state_dict"])
            self.model.load_state_dict(checkpoint["model_state_dict"])
            params = checkpoint["params"]
            params.stability_time += self.params.stability_time
            self.params = params 
            self.params.n_epochs = checkpoint['epoch'] + self.params.n_epochs
            self.params.start_epoch = checkpoint['epoch'] + 1
            self.logger = checkpoint["logger"]  # only if exists!

    def step_triplet_batch(self, indices):
        """ Step the model for a single batch of data and extract embedding triplets"""
        indices = indices.type("torch.LongTensor")
        indices = indices.to(self.device)
        indices = indices.unbind(1)

        ind_i, ind_j, ind_k = indices
        embedding, loc, scale = self.model()

        embedding_i = embedding[ind_i]
        embedding_j = embedding[ind_j]
        embedding_k = embedding[ind_k]

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
        nll = torch.mean(-log_softmax_ij)
    
        # log probability of variational distribution
        log_q = torch.distributions.LogNormal(loc, scale)
        log_p = torch.distributions.LogNormal(self.prior.loc, self.prior.scale)
        kl_div = torch.distributions.kl_divergence(log_q, log_p)


        # log_q1 = torch.distributions.LogNormal(loc, scale).log_prob(embedding)
        # log_p1 = torch.distributions.LogNormal(self.prior.loc, self.prior.scale).log_prob(embedding)
        # kl_div = -torch.sum(log_q1 - log_p1)




        # n_triplets = len(self.train_loader.dataset)
        # kl_div = F.kl_div(log_q1, log_p1, reduction="batchmean", log_target=True)


        return nll, kl_div, triplet_accuracy

    def step_dataloader(self, dataloader):
        n_triplets = len(dataloader.dataset)
        n_batches = len(dataloader)

        complex_losses = torch.zeros(n_batches)
        nll_losses = torch.zeros(n_batches)
        losses = torch.zeros(n_batches)

        triplet_accuracies = torch.zeros(n_batches)

        for k, indices in enumerate(dataloader):

            if self.model.training:
                nll, kl_div, accuracy = self.step_triplet_batch(indices)

                nll = nll - nll


                complexity_loss = kl_div.sum() / n_triplets

                            

                # Balance the loss with the gamma hyperparameter!
                nll = self.params.gamma * nll
                complexity_loss = (1 - self.params.gamma) * complexity_loss
                loss = nll + complexity_loss
            

                # Balance the log likelihood and complexity loss by gamma
                # loss = (self.params.gamma * nll) + ((1 - self.params.gamma) * complexity_loss)

                # faster alternative to optim.zero_grad()
                for param in self.model.parameters():
                    param.grad = None

                loss.backward()
                self.optim.step()

                print(
                    f"Train Batch {k}/{n_batches} NLL {nll.item()}, Complexity {complexity_loss.item()}",
                    end="\r",
                )

            # we do mc sampling of the variational posterior for validation batches
            else:
                sampled_likelihoods = torch.zeros(
                    (self.params.mc_samples, self.params.init_dim)
                )
                sampled_complexities = torch.zeros(
                    (self.params.mc_samples, self.params.init_dim)
                )

                for s in range(self.params.mc_samples):
                    nll, kl_div, accuracy = self.step_triplet_batch(indices)
                    complexity_loss = kl_div.sum() / n_triplets

                    # NOTE For the val loss, we are not allowed to reweight using gamma, 
                    # since we want to compare different models on the same scale!
                    sampled_likelihoods[s] = nll.item()
                    sampled_complexities[s] = complexity_loss.item()

                # validation loss
                loss = torch.mean(sampled_likelihoods) # + torch.mean(sampled_complexities)


                print(
                    f"Val Batch {k}/{n_batches} NLL + Kl Div {loss}",
                    end="\r",
                )

            # def update loss history!
            complex_losses[k] += complexity_loss.item()
            nll_losses[k] += nll.item()
            losses[k] += loss.item()
            triplet_accuracies[k] += accuracy.item()

        loss = losses.mean().item()

        if self.model.training:
            self.params.update(
                train_complexity=complex_losses.mean().item(),
                train_nll=nll_losses.mean().item()
            )
        else:
            self.params.update(
                val_complexity=complex_losses.mean().item(),
                val_nll=nll_losses.mean().item()
            )
        
        epoch_accuracy = torch.mean(triplet_accuracies).item()

        return loss, epoch_accuracy

    def train_one_epoch(self):
        self.model.train(True)
        train_loss = self.step_dataloader(self.train_loader)

        abc = self.model()
        print(abc[0].mean().detach().cpu().numpy())

        return train_loss

    @torch.no_grad()
    def evaluate_one_epoch(self):
        self.model.eval()
        val_loss = self.step_dataloader(self.val_loader)
        return val_loss

    def evaluate_val_loss_convergence(self):
        """ Do we maybe need to analyse converge in term of the validation loss? """
        # NOTE still need to implement this to have an analysis of convergence!
        if self.params.best_val_loss > self.params.val_loss[-1]:
            self.params.update(best_val_loss=self.params.val_loss[-1])

        if (
            self.params.best_val_loss
            not in self.params.val_loss[: -self.params.stability_time]
        ):
            return True

        return False

    def evaluate_convergence(self):
        """ We evaluate convergence as the representational stability of the number of dimensions across a certain time 
        frame of epochs """
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
                    train_complexity=self.params.train_complexity,
                    val_loss=val_loss,
                    val_nll=self.params.val_nll,
                    val_complexity=self.params.val_complexity,
                    val_acc=val_acc,
                    model=self.model,
                    gamma=self.params.gamma,    
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
                    print(f"Stopped training after {self.epoch} epochs. Model has converged or max number of epochs have been reached!")
                    log_params["final"] = True # we also log all things that dont have an update interval
                    self.logger.log(**log_params)
                    self.store_final_embeddings(**log_params)
                    break

                self.logger.log(**log_params)

        except KeyboardInterrupt:
            print("Training interrupted by user. Saving model and exiting.")
            self.logger.log(**log_params)
            self.store_final_embeddings(**log_params)

    def store_final_embeddings(self, **kwargs):
        params = self.model.sorted_pruned_params()
        f_path = os.path.join(self.log_path, "parameters.npz")
        self.params.update(pruned_q_mu=params["pruned_q_mu"], pruned_q_var=params["pruned_q_var"])
        self.params.save(f_path)