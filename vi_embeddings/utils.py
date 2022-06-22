import os
import logging
import sys
import numpy as np
import scipy.stats


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