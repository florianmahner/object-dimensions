#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os
import math

import pandas as pd
import scipy.stats
import numpy as np

from numba import njit, jit, prange
from scipy.spatial.distance import pdist, squareform
from thingsvision import Extractor
from thingsvision.utils.data.dataset import ImageDataset

def load_model(model_name, device):
    model = Extractor(model_name, device=device, pretrained=True, source='torchvision')
    return model

def load_image_data(n_images, modality="deep"):
    if modality == "behavior":
        dataset = ImageDataset(root=f'./data/reference_images', out_path='', backend='pt')
        idx2obj = None # NOTE maybe for behavior these cannot be extracted!
        obj2idx = None

    else:
        dataset = ImageDataset(root=f'./data/image_data/images{n_images}', out_path='', backend='pt')
        idx2obj = dataset.idx_to_cls
        obj2idx = dataset.cls_to_idx

    images = dataset.images

    return idx2obj, obj2idx, images

def filter_embedding_by_behavior(embedding, image_paths):
    behavior_indices = np.array([i for i, img in enumerate(image_paths) if '01b' in img])
    image_paths = np.array(image_paths)[behavior_indices]
    embedding = embedding[behavior_indices]

    return embedding, image_paths

# Determine the cosine similarity between two vectors in pytorch
def cosine_similarity(embedding_i, embedding_j):
    if isinstance((embedding_i, embedding_j), torch.Tensor):
        return torch.nn.functional.cosine_similarity(embedding_i, embedding_j)
    else:
        return np.dot(embedding_i, embedding_j) / (np.linalg.norm(embedding_j) * np.linalg.norm(embedding_j))
    
def compute_positive_rsm(F):
    rsm = relu_correlation_matrix(F)
    return 1 - rsm

def relu_embedding(W):
    return np.maximum(0, W)

def remove_zeros(W, eps=.1):
    w_max = np.max(W, axis=1)
    W = W[np.where(w_max > eps)]
    return W

def get_weights(path):
    W = np.loadtxt(os.path.join(path))
    W = relu_embedding(W)
    W = remove_zeros(W)
    return W

def load_deepnet_activations(activation_path, center=False):
    if activation_path.endswith('npy'):
        with open(activation_path, 'rb') as f:
            act = np.load(f)
    else:
        act = np.loadtxt(activation_path)
    if center:
        center_activations(act)
    return act


def center_activations(act):
    return act - act.mean(axis=0)

def load_sparse_codes(path, with_dim=False):    
    W = get_weights(path)
    sorted_dims = np.argsort(-np.linalg.norm(W, axis=0, ord=1))
    W = W[:, sorted_dims]

    d1, d2 = W.shape
    # We transpose so that the matrix is always of shape (n_images, n_dims)
    if d1 < d2:
        W = W.T

    if with_dim:
        return W, sorted_dims
    else:
        return W

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


@njit(parallel=True, fastmath=True)
def matmul(A, B):
    i, k = A.shape
    k, j = B.shape
    C = np.zeros((i, j))
    for i in prange(i):
        for j in prange(j):
            for k in prange(k):
                C[i, j] += A[i, k] * B[k, j]
    return C

@njit(parallel=True, fastmath=True)
def rsm_pred(W:np.ndarray) -> np.ndarray:
    """convert weight matrix corresponding to the mean of each dim distribution for an object into a RSM"""
    n = W.shape[0]
    S = matmul(W, W.T)
    S_e = np.exp(S) #exponentiate all elements in the inner product matrix S
    rsm = np.zeros((n, n))
    for i in prange(n):
        for j in prange(i+1, n):
            for k in prange(n):
                if (k != i and k != j):
                    rsm[i, j] += S_e[i, j] / (S_e[i, j] + S_e[i, k] + S_e[j, k])
    rsm /= n - 2
    rsm += rsm.T #make similarity matrix symmetric
    return rsm
