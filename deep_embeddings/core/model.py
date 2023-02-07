#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import re
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from deep_embeddings.core.pruning import NormalDimensionPruning, LogNormalDimensionPruning
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


class VariationalEmbedding(nn.Module):
    r"""This is the representational embedding for objects that is specified by a variational
    distribution. The variational distribution is parameterized by a mean and a log-variance.
    The mean and log variance are linear transformations of the input data. Both are free parameters
    that are learned during training."""

    def __init__(self, prior, n_objects, init_dim=100, non_zero_weights=5):
        super().__init__()
        self.prior = prior
        cdf_loc = self.prior.mode

        self.non_zero_weights = non_zero_weights
        self.init_dim = init_dim
        self.n_objects = n_objects

        self.q_mu = QMu(n_objects, init_dim)
        self.q_logvar = QLogVar(n_objects, init_dim)

        self._init_weights()

        if isinstance(prior, LogGaussianPrior):
            # I Think this is the same as pruning using a normal with cdf_loc=0
            self.pruner = LogNormalDimensionPruning(n_objects, cdf_loc=cdf_loc)
        elif isinstance(prior, SpikeSlabPrior):
            self.pruner = NormalDimensionPruning(n_objects, cdf_loc=cdf_loc)

    @staticmethod
    def reparameterize_sslab(loc, scale):
        eps = scale.data.new(scale.size()).normal_()
        return loc + scale * eps

    def reparameterize(self, loc, scale):
        """Apply reparameterization trick."""
        eps = scale.data.new(scale.size()).normal_()
        out = loc + scale * eps

        if isinstance(self.prior, LogGaussianPrior):
            out = out.exp()  # this becomes log normal now

        return out

    def _init_weights(self):
        """Initialize weights for the embedding"""

        if isinstance(self.prior, LogGaussianPrior):
            nn.init.kaiming_normal_(self.q_mu.q_mu, mode="fan_in", nonlinearity="relu")
            eps = -(1 / self.q_mu.q_mu.std())
            self.q_logvar.q_logvar.data.fill_(eps)

        elif isinstance(self.prior, SpikeSlabPrior):
            nn.init.kaiming_normal_(self.q_mu.q_mu, mode="fan_in", nonlinearity="relu")

            eps = -(self.q_mu.q_mu.data.std().log() * -1.0).exp()
            self.q_logvar.q_logvar.data.fill_(eps)

    def forward(self):
        q_mu = self.q_mu()
        q_var = self.q_logvar().exp()  # we need to exponentiate the logvar
        X = self.reparameterize(q_mu, q_var)  # This is a sample from a gaussian

        return X, q_mu, q_var

    @torch.no_grad()
    def prune_dimensions(self, alpha=0.05):
        q_mu = self.q_mu()
        q_var = self.q_logvar().exp()

        importance = self.pruner(q_mu, q_var, alpha=alpha)
        dims_included = torch.where(importance > self.non_zero_weights)[0]

        pruned_loc = q_mu[:, dims_included]
        pruned_scale = q_var[:, dims_included]

        return dims_included, pruned_loc, pruned_scale

    def detached_params(self):
        """Detach params from computational graph"""
        # TODO Add argsort maybe
        q_mu = self.q_mu().detach()
        q_var = self.q_logvar().exp().detach()
        params = dict(q_mu=q_mu.cpu().numpy(), q_var=q_var.cpu().numpy())

        return params

    def sorted_pruned_params(self):
        _, pruned_loc, pruned_scale = self.prune_dimensions()
        argsort_loc = torch.argsort(
            torch.linalg.norm(pruned_loc, dim=0, ord=1), descending=True
        )
        pruned_loc = pruned_loc[:, argsort_loc]
        pruned_scale = pruned_scale[:, argsort_loc]

        func = (
            torch.distributions.LogNormal
            if isinstance(self.prior, LogGaussianPrior)
            else torch.distributions.Normal
        )
        embedding = func(pruned_loc, pruned_scale).sample()

        embedding = embedding.detach().cpu().numpy()
        pruned_loc = pruned_loc.detach().cpu().numpy()
        pruned_scale = pruned_scale.detach().cpu().numpy()
        params = dict(
            pruned_q_mu=pruned_loc, pruned_q_var=pruned_scale, embedding=embedding
        )

        return params


class DeterministicEmbedding(nn.Module):
    r"""This is the representational embedding for objects for SPoSE. The embedding is specified by a point-estimate of our posterior
    distribution where only the mean is used. The mean is a linear transformation of the input data. It is a free parameter
    that is learned during training."""
    def __init__(self, n_objects, init_dim, init_weights=True):
        super().__init__()
        self.n_objects = n_objects
        self.init_dim = init_dim
        self.fc = nn.Linear(self.init_dim, self.n_objects, bias=False)
        if init_weights:
            self._initialize_weights()

    def forward(self):
        return self.fc.weight

    def _initialize_weights(self):
        mean, std = .1, .01
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, std)

    def l1_regularization(self):
        l1_reg = torch.tensor(0., requires_grad=True)
        for n, p in self.named_parameters():
            if re.search(r'weight', n):
                l1_reg = l1_reg + torch.norm(p, 1)
        return l1_reg

    def get_nneg_dims(self, weights, eps=0.1):
        w_max = weights.max(0)[0]
        nneg_d = len(w_max[w_max > eps])
        
        return nneg_d

    def sort_weights(self):
        weights = self.fc.weight
        index = torch.argsort(
            torch.linalg.norm(weights, dim=0, ord=1), descending=True
        )
        weights = weights[:, index]

    def detached_params(self):
        weights = self.sort_weights()
        weights = weights.detach().cpu().numpy()

        return dict(weights=weights)

    def sorted_pruned_params(self):
        weights = self.prune_dimensions()[1]
        weights = weights.detach().cpu().numpy()
        params = dict(pruned_weights=weights)

    @torch.no_grad()
    def prune_dimensions(self, alpha=0.05):
        weights = self.fc.weight
        index = torch.argsort(
            torch.linalg.norm(weights, dim=0, ord=1), descending=True
        )
        weights = weights[:, index]
        eps = 0.1
        w_max = weights.max(0)[0]
        dims_included = w_max > eps
        weights = weights[:, dims_included]

        return dims_included, weights