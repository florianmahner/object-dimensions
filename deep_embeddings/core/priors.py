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
    def log_pdf(self):
        """ Probability density function of the prior """
        pass


class LogGaussianPrior(BasePrior):
    def __init__(self, n_objects, init_dim, loc=0., scale=0.5):
        super().__init__()
        
        self.register_buffer("loc", torch.zeros(n_objects, init_dim) + loc) 
        self.register_buffer("scale", torch.ones(n_objects, init_dim) * scale)
        
    def log_pdf(self, X, loc, scale):
        """ Calculate the probability density function of the log-normal distribution """
        pdf = torch.distributions.LogNormal(loc, scale).log_prob(X)

        return pdf

    @property
    def mode(self):
        """ Calculate the mode of the log-normal distribution """
        return torch.exp(self.loc)

    @property
    def mean(self):
        """ Mean of the log-normal distribution """
        return self.loc[0,0]

    @property
    def variance(self):
        return self.scale[0,0]
    
    def forward(self, X):
        return self.log_pdf(X, self.loc, self.scale)

    



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

    def log_pdf(self, X, loc, scale):
        log_pdf = torch.distributions.Normal(loc, scale).log_prob(X)

        return log_pdf

    def forward(self, X):
        spike = self.pi * torch.distributions.Normal(self.loc, self.spike).log_prob(X)
        slab = (1 - self.pi) * torch.distributions.Normal(self.loc, self.slab).log_prob(X)
        
        return spike + slab
    

    @property
    def mode(self):
        return torch.tensor(0)