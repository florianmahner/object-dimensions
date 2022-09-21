#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os
import math
import scipy.stats

import numpy as np
from scipy.spatial.distance import pdist, squareform


# Determine the cosine similarity between two vectors in pytorch
def cosine_similarity(embedding_i, embedding_j):
    if isinstance((embedding_i, embedding_j), torch.Tensor):
        return torch.nn.functional.cosine_similarity(embedding_i, embedding_j)
    else:
        return np.dot(embedding_i, embedding_j) / (np.linalg.norm(embedding_j) * np.linalg.norm(embedding_j))

def compute_positive_rsm(F):
    rsm = relu_correlation_matrix(F)
    return 1 - rsm

def remove_zeros(W, eps=.1):
    w_max = np.max(W, axis=1)
    W = W[np.where(w_max > eps)]
    return W

def get_weights(path):
    W = np.loadtxt(os.path.join(path))
    return remove_zeros(W)

def load_sparse_codes(path):
    W = get_weights(path)
    l1_norms = np.linalg.norm(W, ord=1, axis=1)
    sorted_dims = np.argsort(l1_norms)[::-1]
    W = W[sorted_dims]
    return W, sorted_dims

def fill_diag(rsm):
    """ Fill main diagonal of the RSM with ones """
    assert np.allclose(rsm, rsm.T), '\nRSM is required to be a symmetric matrix\n'
    rsm[np.eye(len(rsm)) == 1.] = 1

    return rsm

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

def compute_rdm(X, method="correlation"):
    assert method in ["correlation", "euclidean"]
    if method == "euclidean":
        rdm = squareform(pdist(X, metric="euclidean"))        
    else:
        rsm = correlation_matrix(X)
        rdm = 1 - rsm

    return rdm

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

# NOTE This seems to be wrong -> not the same results as obtained from torch.Distributions or scipy.stats
# def normal_pdf(X, loc, scale):
    # gauss_pdf = torch.exp(-((X - loc) ** 2) / (2 * scale.pow(2))) / scale * math.sqrt(2 * math.pi)

    # return gauss_pdf

def normal_pdf(X, loc, scale):
    """Probability density function of a normal distribution."""
    return (
        torch.exp(-((X - loc) ** 2) / (2 * scale.pow(2)))
        / scale
        * math.sqrt(2 * math.pi)
    )

def log_normal_pdf(X, loc, scale):
    return torch.distributions.Normal(loc, scale).log_prob(X)
