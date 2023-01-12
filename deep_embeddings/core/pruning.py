#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from functools import partial


class NormalDimensionPruning(nn.Module):
    r"""We prune dimensions based on the importance of each dimension, which is determined by the set of pvalues obtained from the
    cdf of the normal distribution given the mean and variance of the embedding, corrected for multiple comparisons by calculting
    the FDR given the Benjamin-Hochberg procedure."""

    def __init__(self, n_objects, cdf_loc=0.0):
        super().__init__()
        if not isinstance(cdf_loc, torch.Tensor):
            cdf_loc = torch.tensor(cdf_loc)
        self.register_buffer("cdf_loc", cdf_loc)
        # empirical cumulative distribution function factor for the benjamin hochberg correction
        self.register_buffer("ecdf_factor", self._ecdf_torch(n_objects)) 

    def __call__(self, q_mu, q_var, alpha=0.05):
        pvals = self.compute_pvals_torch(q_mu, q_var)
        rejections = self.adjust_pvals_mutliple_comparisons_torch(pvals, alpha)
        importance = self.get_importance_torch(rejections)
        return importance

    def pval_torch(self, q_mu, q_var, j):
        # the cdf describes the probability that a random sample X of n objects at dimension j
        # will be less than or equal to 0 (in our case) for a given mean (mu) and standard deviation (sigma):
        return torch.distributions.Normal(q_mu[:, j], q_var[:, j]).cdf(self.cdf_loc)
        
    def compute_pvals_torch(self, q_mu, q_var):
        # we compute the cdf probabilities >0 for all dimensions
        fn = partial(self.pval_torch, q_mu, q_var)
        n_dim = q_mu.shape[1]
        range_dim = torch.arange(n_dim)
        pvals = fn(range_dim)

        return pvals.T

    def adjust_pvals_mutliple_comparisons_torch(self, p_vals, alpha=0.05):
        fdr = torch.empty_like(p_vals)
        n_pvals = p_vals.shape[0]
        for i in range(n_pvals):
            fdr[i] = self.fdr_correction_torch(p_vals[i], alpha=alpha)[0]
        return fdr

    def get_importance_torch(self, rejections):
        importance = rejections.sum(dim=1)
        return importance

    def _ecdf_torch(self, nobs):
        """No frills empirical cdf used in fdrcorrection"""
        return torch.arange(1, nobs + 1, dtype=torch.float64) / float(nobs)

    def fdr_correction_torch(self, pvals, alpha=0.05, is_sorted=False):
        """Pytorch implementation of fdr correction for only the benjamin hochberg method. see for more details:
        https://github.com/statsmodels/statsmodels/blob/09a89a3b11445e6221eba87a1f86a6789a725be2/statsmodels/stats/multitest.py#L63-L266"""

        assert pvals.ndim == 1, "pvals must be 1-dimensional, that is of shape (n,)"
        if not is_sorted:
            pvals_sortind = torch.argsort(pvals)
            pvals_sorted = torch.take(pvals, pvals_sortind)
        else:
            pvals_sorted = pvals

        reject = pvals_sorted <= self.ecdf_factor * alpha

        if reject.any():
            rejectmax = max(torch.nonzero(reject)[0])
            reject[:rejectmax] = True

        pvals_corrected_raw = pvals_sorted / self.ecdf_factor

        # torch.cummin is the same as np.minimum.accumulate
        # torch.flip(a, dims=(0,)) is the same as a[::-1]
        pvals_cummin, _ = torch.cummin(
            torch.flip(pvals_corrected_raw, dims=(0,)), dim=0
        )
        pvals_corrected = torch.flip(pvals_cummin, dims=(0,))
        del pvals_corrected_raw
        pvals_corrected[pvals_corrected > 1] = 1

        if not is_sorted:
            pvals_corrected_ = torch.empty_like(pvals_corrected)
            pvals_corrected_[pvals_sortind] = pvals_corrected
            del pvals_corrected
            reject_ = torch.empty_like(reject)
            reject_[pvals_sortind] = reject
            return reject_, pvals_corrected_
        else:
            return reject, pvals_corrected
