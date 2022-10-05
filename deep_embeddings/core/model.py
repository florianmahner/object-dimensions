#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

from deep_embeddings.pruning import DimensionPruning

                
class QLogVar(nn.Module):
    ''' Log variance of the variational distribution q '''
    def __init__(self, n_objects, init_dim=100, bias=True):
        super().__init__()
        self.q_logvar = nn.Linear(n_objects, init_dim, bias=bias)

    def forward(self):
        return self.q_logvar.weight.T


class QMu(nn.Module):
    ''' Mean of the variational distribution q '''
    def __init__(self, n_objects, init_dim=100, bias=True):
        super().__init__()
        self.q_mu = nn.Linear(n_objects, init_dim, bias=bias)
        nn.init.kaiming_normal_(self.q_mu.weight, mode='fan_out', nonlinearity='relu')

    def forward(self):
        return self.q_mu.weight.T


class VI(nn.Module):
    def __init__(self, n_objects, init_dim=100, init_weights=True):
        super().__init__()
        self.q_mu = QMu(n_objects, init_dim, bias=True) 
        self.q_logvar = QLogVar(n_objects, init_dim, bias=True)
        self.non_zero_weights = 5 # minimum number of non-zero weights
        self.init_dim = init_dim
        if init_weights:
            self._initialize_weights()

        self.pruner = DimensionPruning(n_objects)

    @staticmethod
    def reparameterize(loc, scale):
        """Apply reparameterization trick."""
        eps = scale.data.new(scale.size()).normal_()
        return eps.mul(scale).add(loc)

    @staticmethod
    def reparameterize_old(q_mu, q_var):
        ''' Apply reparameterization trick '''             
        # scale = torch.exp(0.5 * q_logvar) + 1e-5 NOTE i dont need this?!
        eps = torch.randn_like(q_var)
        X = q_mu + q_var * eps

        return X

    def _initialize_weights(self):
        # this is equivalent to - (1 / std(mu)) = -std(mu)^{-1}
        eps = -(self.q_mu.q_mu.weight.data.std().log() * -1.0).exp()
        nn.init.constant_(self.q_logvar.q_logvar.weight, eps)

    def forward(self):
        q_mu = self.q_mu()
        q_var = self.q_logvar().exp() # we need to exponentiate the logvar
        X = self.reparameterize(q_mu, q_var)

        # TODO Maybe do F.relu(X)?, z = F.relu(X) -> we need original, not relu X for prior eval.
        
        return X, q_mu, q_var

    @torch.no_grad()
    def prune_dimensions(self, alpha=0.05):
        q_mu = self.q_mu()
        q_var = self.q_logvar().exp()

        importance = self.pruner(q_mu, q_var, alpha)
        signal = torch.where(importance > self.non_zero_weights)[0] # we have a certain minimum number of dimensions that we maintain!
        pruned_loc = q_mu[:, signal]
        pruned_scale = q_var[:, signal]

        return signal, pruned_loc, pruned_scale

    def detached_params(self):
        ''' Detach params from computational graph ''' 
        q_mu = self.q_mu().detach()
        q_var = self.q_logvar().exp().detach()
        params = dict(q_mu=q_mu.cpu().numpy(), q_var=q_var.cpu().numpy())
        
        return params

    def sorted_pruned_params(self):
        _, pruned_loc, pruned_scale = self.prune_dimensions()
        pruned_loc = pruned_loc.detach().cpu().numpy()
        pruned_scale = pruned_scale.detach().cpu().numpy()
        pruned_loc = np.maximum(0, pruned_loc) # we really zero out all negative values in the embedding!
        pruned_scale = np.maximum(0, pruned_scale)
        pruned_loc = pruned_loc[:, np.argsort(-np.linalg.norm(pruned_loc, axis=0, ord=1))]
        pruned_scale = pruned_scale[:, np.argsort(-np.linalg.norm(pruned_loc, axis=0, ord=1))]
        params = dict(pruned_q_mu=pruned_loc, pruned_q_var=pruned_scale)
        return params