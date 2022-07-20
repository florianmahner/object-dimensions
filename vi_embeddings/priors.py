import torch
import vi_embeddings.utils as utils
import torch.nn as nn

class WeibullPrior(nn.Module):
    def __init__(self, lmbda=1., k=0.5):
        super().__init__()
        self.register_buffer('lmbda', torch.tensor(lmbda)) 
        self.register_buffer('k', torch.tensor(k))

        # with lambda=1, k=0.5 this is the same as an exponential distribution!
        self.distrib = torch.distributions.Weibull(self.lmbda, self.k)

    def forward(self, X):
        return self.distrib.log_prob(X).exp()


     
class ExponentialPrior(nn.Module):
    def __init__(self, n_objects, lmbda=0.5):
        super().__init__()
        self.register_buffer('lmbda', torch.tensor(lmbda)) 
        self.n_objects = n_objects

    def forward(self, X):
        # NOTE temporarily remove the - sign from the exponent! -> this then somewhat works????
        proba =  self.lmbda * torch.exp(self.lmbda * X)

        return proba


class SpikeSlabPrior(nn.Module):
    def __init__(self, n_objects, init_dim, pi=0.5, spike=0.25, slab=1.0):
        super().__init__()
        self.register_buffer('loc', torch.zeros(n_objects, init_dim)) 
        self.register_buffer('pi', torch.ones(n_objects, init_dim) * pi)
        self.register_buffer('spike', torch.ones(n_objects, init_dim) * spike)
        self.register_buffer('slab', torch.ones(n_objects, init_dim) * slab)

    def spike_and_slab(self, X):
        spike = self.pi * utils.normalized_pdf(X, self.loc, self.spike)
        slab = (1 - self.pi) * utils.normalized_pdf(X, self.loc, self.slab)
        
        return spike + slab

    def forward(self, X):
        return self.spike_and_slab(X)


class GaussianPrior(nn.Module):
    def __init__(self, n_objects, init_dim=100):
        super().__init__()
        self.register_buffer('loc', (torch.zeros(n_objects, init_dim)))
        self.register_buffer('scale', (torch.ones(n_objects, init_dim)))

    def __call__(self, X):
        return utils.normalized_pdf(X, self.loc, self.scale)