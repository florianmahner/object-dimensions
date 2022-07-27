import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import vi_embeddings.utils as utils

                
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
        self.non_zero_weights = 10 # minimum number of non-zero weights
        self.init_dim = init_dim
        if init_weights:
            self._initialize_weights()

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

        # NOTE very important question: do i need to do the gradient here to have it differentiable?!
        # probably not -> maybe also dont make the embedding positive, since the positvity constraint then doesnt work?
        # q_mu = F.relu(q_mu)
        
        return X, q_mu, q_var

    def prune_dimensions(self, alpha=0.05):
        # Prune the variational parameters theta
        # by identifying the number of *relevant* dimensions
        # according to our dimensionality reduction procedure,
        # defined in VICE §3.3.4
        q_mu = self.detached_params()['q_mu']
        q_var = self.detached_params()['q_var']
        p_vals = utils.compute_pvals(q_mu, q_var)
        rejections = utils.fdr_corrections(p_vals, alpha)
        importance = utils.get_importance(rejections).ravel()
        signal = np.where(importance > self.non_zero_weights)[0]
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
        pruned_loc = pruned_loc[:, np.argsort(-np.linalg.norm(pruned_loc, axis=0, ord=1))]
        pruned_scale = pruned_scale[:, np.argsort(-np.linalg.norm(pruned_loc, axis=0, ord=1))]
        params = dict(pruned_q_mu=pruned_loc, pruned_q_var=pruned_scale)
        return params
