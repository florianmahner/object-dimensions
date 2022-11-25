#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from deep_embeddings.core.pruning_lognormal import LogNormalDimensionPruning
from deep_embeddings.core.pruning import NormalDimensionPruning

from deep_embeddings.core.priors import SpikeSlabPrior, LogGaussianPrior


class QLogVar(nn.Module):
    """Log variance of the variational distribution q"""

    def __init__(self, n_objects, init_dim):
        super().__init__()
        self.q_logvar = nn.Parameter(torch.FloatTensor(n_objects, init_dim))

    def forward(self):
        return self.q_logvar


class QMu(nn.Module):
    """Mean of the variational distribution q"""

    def __init__(self, n_objects, init_dim):
        super().__init__()
        self.q_mu = nn.Parameter(torch.FloatTensor(n_objects, init_dim))

    def forward(self):
        return self.q_mu


class Embedding(nn.Module):
    r"""This is the representational embedding for objects. It is specified by a variational
    distribution. The variational distribution is parameterized by a mean and a log-variance.
    The mean and log variance are linear transformations of the input data. Both are free parameters
    that are learned during training."""

    def __init__(self, prior, n_objects, init_dim=100, non_zero_weights=5):
        super().__init__()
        self.prior = prior
        cdf_loc = self.prior.mode

        self.non_zero_weights = non_zero_weights
        self.init_dim = init_dim

        self.q_mu = QMu(n_objects, init_dim)
        self.q_logvar = QLogVar(n_objects, init_dim)

        self._init_weights()

        if isinstance(prior, LogGaussianPrior):
            self.pruner = LogNormalDimensionPruning(n_objects, cdf_loc=cdf_loc)

        elif isinstance(prior, SpikeSlabPrior):
            self.pruner = NormalDimensionPruning(n_objects, cdf_loc=cdf_loc)

    @staticmethod
    def reparameterize_sslab(loc, scale):
        """Apply reparameterization trick."""
        eps = scale.data.new(scale.size()).normal_()

        return loc + scale * eps

    def reparameterize(self, loc, scale):
        """Apply reparameterization trick."""

        if isinstance(self.prior, LogGaussianPrior):
            # eps = nn.init.trunc_normal_(scale.data.new(scale.size()), mean=0, std=1, a=0.0)
            eps = scale.data.new(scale.size()).log_normal_(0, 1)

        elif isinstance(self.prior, SpikeSlabPrior):
            eps = scale.data.new(scale.size()).normal_()

        return loc + scale * eps

    def _init_weights(self):
        """Initialize weights for the embedding"""

        if isinstance(self.prior, LogGaussianPrior):
            # nn.init.trunc_normal_(self.q_mu.q_mu.data, mean=0, std=1, a=0.0)
            self.q_mu.q_mu.data.log_normal_(
                mean=self.prior.mean, std=self.prior.variance
            )
            # self.q_mu.q_mu.data.log_normal_(mean=0, std=1)
            eps2 = -(self.q_mu.q_mu.data.std().log() * -1.0).exp()
            self.q_logvar.q_logvar.data.fill_(eps2)

            # nn.init.kaiming_normal_(self.q_mu.q_mu.data, mode="fan_out", nonlinearity="relu")
            # eps = self.q_mu.q_mu.std().log()
            # self.q_logvar.q_logvar.data.fill_(eps)

            # self.q_mu.q_mu.data = torch.abs(self.q_mu.q_mu.data)
            # nn.init.uniform_(self.q_logvar.q_logvar.data, a=-2.5, b=-2.0)
            # self.q_logvar.q_logvar.data.normal_(mean=-2.5, std=0.001)

        elif isinstance(self.prior, SpikeSlabPrior):
            nn.init.kaiming_normal_(
                self.q_mu.q_mu.data, mode="fan_out", nonlinearity="relu"
            )
            eps = self.q_mu.q_mu.std().log()
            self.q_logvar.q_logvar.data.fill_(eps)

            # eps2 = -(self.q_mu.q_mu.data.std().log() * -1.0).exp()
            # self.q_logvar.q_logvar.data.fill_(eps2)

    def forward(self):
        q_mu = self.q_mu()
        q_var = self.q_logvar().exp()  # we need to exponentiate the logvar

        # This is a normal distribution
        X = self.reparameterize(q_mu, q_var)

        if isinstance(self.prior, LogGaussianPrior):
            X = F.relu(X) + 1e-12

        return X, q_mu, q_var

    @torch.no_grad()
    def prune_dimensions(self, alpha=0.05):
        q_mu = self.q_mu()
        q_var = self.q_logvar().exp()
        importance = self.pruner(q_mu, q_var, alpha=alpha)
        signal = torch.where(importance > self.non_zero_weights)[0]

        pruned_loc = q_mu[:, signal]
        pruned_scale = q_var[:, signal]

        return signal, pruned_loc, pruned_scale

    def detached_params(self):
        """Detach params from computational graph"""
        q_mu = self.q_mu().detach()
        q_var = self.q_logvar().exp().detach()
        params = dict(q_mu=q_mu.cpu().numpy(), q_var=q_var.cpu().numpy())

        return params

    def sorted_pruned_params(self):
        _, pruned_loc, pruned_scale = self.prune_dimensions()
        pruned_loc = pruned_loc.detach().cpu().numpy()
        pruned_scale = pruned_scale.detach().cpu().numpy()
        argsort_loc = np.argsort(-np.linalg.norm(pruned_loc, axis=0, ord=1))
        pruned_loc = pruned_loc[:, argsort_loc]
        pruned_scale = pruned_scale[:, argsort_loc]
        params = dict(pruned_q_mu=pruned_loc, pruned_q_var=pruned_scale)

        return params
