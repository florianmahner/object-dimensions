import torch
import torch.nn as nn
import math
from functools import partial

class Pruning:

    def __init__(self):
        pass

    def compute_pvals(self, loc, scale):
        ''' NOTE Maybe we make different assumptions here if we assume gaussian features with zero mean
        and unit var, since the positivity constraint is then a different one? Have to check this! '''
        pass

    def fdr_correct(self):
        pass

    def get_importance(self):
        pass





class GaussianPrior(nn.Module):
    def __init__(self, n_objects, init_dim=100):
        super().__init__()
        self.register_buffer('loc', (torch.zeros(n_objects, init_dim)))
        self.register_buffer('scale', (torch.ones(n_objects, init_dim)))

    @staticmethod
    def normalized_pdf(X, loc, scale):
        gauss_pdf = torch.exp(-((X - loc) ** 2) / (2 * scale.pow(2))) / scale * math.sqrt(2 * math.pi)

        return gauss_pdf

    def __call__(self, X):
        return self.normalized_pdf(X, self.loc, self.scale)

                
class QLogVar(nn.Module):
    ''' Log variance of the variational distribution q '''
    def __init__(self, n_objects, init_dim=100, bias=True):
        super().__init__()
        self.q_logvar = nn.Linear(n_objects, init_dim, bias=bias)

    def forward(self):
        return self.q_logvar.weight.T.exp()


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
        if init_weights:
            self._initialize_weights()

    @staticmethod
    def reparameterize(q_mu, q_logvar):
        ''' Apply reparameterization trick '''             
        # scale = torch.exp(0.5 * q_logvar) + 1e-5 NOTE dont this i need this!
        eps = torch.randn_like(q_logvar)
    
        return q_mu + q_logvar * eps

    def forward(self):
        q_mu = self.q_mu()
        q_logvar = self.q_logvar()
        X = self.reparameterize(q_mu, q_logvar)
        
        return X, q_mu, q_logvar

    def _initialize_weights(self):
        # this is equivalent to - (1 / std(mu)) = -std(mu)^{-1}
        eps = -(self.q_mu.q_mu.weight.data.std().log() * -1.0).exp()
        nn.init.constant_(self.q_logvar.q_logvar.weight, eps)

    def detached_params(self):
        ''' Detach params from computational graph ''' 
        loc = self.q_mu().detach()
        scale = self.q_logvar().exp().detach()
        params = dict(q_mu=loc.cpu().numpy(), q_var=scale.cpu().numpy())
        
        return params