#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from deep_embeddings.utils import utils
from abc import ABC, abstractmethod


class BasePrior(nn.Module, ABC):
    """ Define an abstract prior class """
    @abstractmethod
    def pdf(self):
        """ Probability density function of the prior """
        pass


class LogGaussianPrior(torch.distributions.LogNormal):
    def __init__(self, loc=0., scale=0.5):
        super().__init__(loc, scale)

    def pdf(self, X):
        """ Calculate the probability density function of the log-normal distribution """
        pdf = self.log_prob(X)

        return pdf.exp()

    def forward(self, X):
        return self.pdf(X)


class SpikeSlabPrior(BasePrior):
    def __init__(self, n_objects, init_dim, pi=0.5, spike=0.25, slab=1.0):
        super().__init__()
        self.register_buffer('loc', torch.zeros(n_objects, init_dim)) 
        self.register_buffer('pi', torch.ones(n_objects, init_dim) * pi)
        self.register_buffer('spike', torch.ones(n_objects, init_dim) * spike)
        self.register_buffer('slab', torch.ones(n_objects, init_dim) * slab)

    def pdf(self, X):
        spike = self.pi * torch.distributions.Normal(self.loc, self.spike).log_prob(X).exp()
        slab = (1 - self.pi) * torch.distributions.Normal(self.loc, self.slab).log_prob(X).exp()
        
        return spike + slab

    def forward(self, X):
        return self.pdf(X)

    @property
    def mode(self):
        return torch.tensor(0)