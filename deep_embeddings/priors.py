import torch
import deep_embeddings.utils as utils
import torch.nn as nn
import torch.nn.functional as F

class WeibullPrior(nn.Module):
    def __init__(self, lmbda=1., k=0.5):
        super().__init__()
        self.register_buffer('lmbda', torch.tensor(lmbda)) 
        self.register_buffer('k', torch.tensor(k))
        self.register_buffer('eps', torch.tensor(1e-12))


    def pdf(self, X):
        const = (self.k / self.lmbda)
        term1 = ((X/self.lmbda) ** (self.k-1))
        term2 = torch.exp(-(X / self.lmbda)**self.k)

        # NOTE same as 
        # pdf = (self.k / self.lmbda) * ((X/self.lmbda) ** (self.k - 1)) * torch.exp((-X / self.lmbda)**self.k)
        pdf = const * term1 * term2

        return pdf

    def forward(self, X):
        return self.pdf(X)

     
class ExponentialPrior(nn.Module):
    def __init__(self, n_objects, lmbda=0.5):
        super().__init__()
        self.register_buffer('lmbda', torch.tensor(lmbda)) 
        self.n_objects = n_objects

    def forward(self, X):
        # NOTE temporarily remove the - sign from the exponent! -> this then somewhat works????
        # proba =  self.lmbda * torch.exp(self.lmbda * X)
        proba =  self.lmbda * torch.exp(-self.lmbda * X)

        return proba


class SpikeSlabPrior(nn.Module):
    def __init__(self, n_objects, init_dim, pi=0.5, spike=0.25, slab=1.0):
        super().__init__()
        self.register_buffer('loc', torch.zeros(n_objects, init_dim)) 
        self.register_buffer('pi', torch.ones(n_objects, init_dim) * pi)
        self.register_buffer('spike', torch.ones(n_objects, init_dim) * spike)
        self.register_buffer('slab', torch.ones(n_objects, init_dim) * slab)

    def spike_and_slab(self, X):
        spike = self.pi * utils.log_normal_pdf(X, self.loc, self.spike).exp()
        slab = (1 - self.pi) * utils.log_normal_pdf(X, self.loc, self.slab).exp()
        
        return spike + slab

    def forward(self, X):
        return self.spike_and_slab(X)

class SpikeSlabPriorConstrained(nn.Module):
    def __init__(self, n_objects, init_dim, pi=0.5, spike=0.25, slab=1.0):
        super().__init__()
        self.register_buffer('loc', torch.zeros(n_objects, init_dim)) 
        self.register_buffer('pi', torch.ones(n_objects, init_dim) * pi)
        self.register_buffer('spike', torch.ones(n_objects, init_dim) * spike)
        self.register_buffer('slab', torch.ones(n_objects, init_dim) * slab)

    def spike_and_slab(self, X, indices):
        loc = self.loc[indices]
        spike = self.spike[indices]
        slab = self.slab[indices]
        pi = self.pi[indices]

        spike = pi * utils.log_normal_pdf(X, loc, spike).exp()
        slab = (1 - pi) * utils.log_normal_pdf(X, loc, slab).exp()
        
        return spike + slab

    def forward(self, X, indices):
        return self.spike_and_slab(X, indices)


class GaussianPrior(nn.Module):
    def __init__(self, n_objects, init_dim=100):
        super().__init__()
        self.register_buffer('loc', (torch.zeros(n_objects, init_dim)))
        self.register_buffer('scale', (torch.ones(n_objects, init_dim)))

    def __call__(self, X):
        return utils.log_normal_pdf(X, self.loc, self.scale).exp()

