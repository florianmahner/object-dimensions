import numpy as np
import scipy.stats
from scipy.stats import norm
from functools import partial
from statsmodels.stats.multitest import multipletests
import torch
import math

# Determine the cosine similarity between two vectors in pytorch
def cosine_similarity(embedding_i, embedding_j):
    if isinstance((embedding_i, embedding_j), torch.Tensor):
        return torch.nn.functional.cosine_similarity(embedding_i, embedding_j)
    else:
        return np.dot(embedding_i, embedding_j) / (np.linalg.norm(embedding_j) * np.linalg.norm(embedding_j))

def compute_positive_rsm(F):
    rsm = relu_correlation_matrix(F)
    return 1 - rsm

def relu_correlation_matrix(F, a_min= -1., a_max= 1.):
    ''' Compute dissimilarity matrix based on correlation distance (on the matrix-level). '''
    F_c = F - F.mean(axis=1)[:, np.newaxis]
    cov = F_c @ F_c.T
    cov = np.maximum(cov, 0)
    # compute vector l2-norm across rows
    l2_norms = np.linalg.norm(F_c, axis=1)
    denom = np.outer(l2_norms, l2_norms)
    corr_mat = (cov / denom).clip(min=a_min, max=a_max)
    
    return corr_mat


def compute_rsm(F):
    rsm = correlation_matrix(F)
    return 1 - rsm

def correlation_matrix(F, a_min= -1., a_max= 1.):
    ''' Compute dissimilarity matrix based on correlation distance (on the matrix-level). '''
    F_c = F - F.mean(axis=1)[:, np.newaxis]
    cov = F_c @ F_c.T
    # compute vector l2-norm across rows
    l2_norms = np.linalg.norm(F_c, axis=1)
    denom = np.outer(l2_norms, l2_norms)
    corr_mat = (cov / denom).clip(min=a_min, max=a_max)
    
    return corr_mat

def correlate_rsms(rsm_a, rsm_b, correlation = 'pearson'):
    triu_inds = np.triu_indices(len(rsm_a), k=1)
    corr_func = getattr(scipy.stats, ''.join((correlation, 'r')))
    rho = corr_func(rsm_a[triu_inds], rsm_b[triu_inds])[0]
    
    return rho

def normalized_pdf(X, loc, scale):
    gauss_pdf = torch.exp(-((X - loc) ** 2) / (2 * scale.pow(2))) / scale * math.sqrt(2 * math.pi)

    return gauss_pdf


def pval(W_loc, W_scale, j):
# the cdf describes the probability that a random sample X of n objects at dimension j 
# will be less than or equal to 0 in our case) for a given mean (mu) and standard deviation (sigma):
    return norm.cdf(0.0, W_loc[:, j], W_scale[:, j])

def compute_pvals(q_mu, q_var):
    # Compute the probability for an embedding value x_{ij} <= 0,
    # given mu and sigma of the variational posterior q_{\theta}
        
    # we compute the cdf probabilities >0 for all dimensions
    fn = partial(pval, q_mu, q_var)
    n_dim = q_mu.shape[1]
    range_dim = np.arange(n_dim)
    pvals = fn(range_dim)

    return pvals.T

def fdr_corrections(p_vals, alpha = 0.05):
    # For each dimension, statistically test how many objects have non-zero weight and account for multiple comparisons
    pval_rejection = lambda p: multipletests(p, alpha=alpha, method='fdr_bh')[0] # true for hypothesis that can be rejected for given alpha
    fdr = np.empty_like(p_vals)
    n_pvals = p_vals.shape[0]
    for i in range(n_pvals):
        fdr[i] = pval_rejection(p_vals[i])

    return fdr
 
def get_importance(rejections):
    # Taken from LukasMut/VICE/utils.py    
    # Yield the the number of rejections given by the False Discovery Rates
    importance = rejections.sum(dim=1)
    
    return importance