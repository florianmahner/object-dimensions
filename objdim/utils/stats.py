#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import math
import numpy as np
from typing import Union
from scipy.stats import pearsonr
from torch.nn import functional as F

def vectorized_pearsonr(base: np.ndarray, comp: np.ndarray) -> Union[float, np.ndarray]:
    """Alterntive to scipy.stats.pearsonr that is vectorized over the first dimension for
    fast pairwise correlation calculation."""
    if base.shape != comp.shape:
        raise ValueError(
            "Input arrays must have the same dimensions; "
            f"base.shape = {base.shape}, comp.shape = {comp.shape}"
        )
    if base.ndim < 2:
        base = base[:, None]
    if comp.ndim < 2:
        comp = comp[:, None]
    n = base.shape[1]
    covariance = np.cov(base.T, comp.T, ddof=1)
    base_std = np.sqrt(covariance[:n, :n].diagonal())
    comp_std = np.sqrt(covariance[n:, n:].diagonal())
    pearson_r = covariance[:n, n:] / np.outer(base_std, comp_std)
    return pearson_r


def normal_pdf(X, loc, scale):
    """Probability density function of a normal distribution."""
    return (
        torch.exp(-((X - loc) ** 2) / (2 * scale.pow(2)))
        / scale
        * math.sqrt(2 * math.pi)
    )


def log_normal_pdf(X, loc, scale):
    LOG_NORMAL_CONST = torch.sqrt(torch.tensor(2 * math.pi))
    """Calculate the probability density function of the log-normal distribution"""
    X = F.relu(X) + 1e-12
    const = 1 / (X * scale * LOG_NORMAL_CONST)
    numerator = -((torch.log(X) - loc) ** 2)
    denominator = 2 * scale**2
    pdf = const * torch.exp(numerator / denominator)

    return pdf


def pairwise_correlate_dimensions(
    weights_human,
    weights_dnn,
    base="human",
    duplicates=False,
    sort_by_corrs=True,
    return_corrs=True,
):
    """Correlate the weights of two modalities and return the weights of both modalities in eiter the same or different orders
    Parameters:
    weights_human (np.ndarray): The weights of the human modality
    weights_dnn (np.ndarray): The weights of the DNN modality
    base (str): The modality that will be used as the base for the comparison.
                The other modality will be compared to this one.
    duplicates (bool): Whether to allow duplicate dimensions in the comparison modality.
                       If set to False, we correlate without repeats.
    sort_by_corrs (bool): Whether to sort the dimensions based on the highest correlations. Otherwise we sort
                            the dimensions based on the order of the dimensions in the base modality
                            (i.e. the sum of the weights)
    Returns:
    Tuple[np.ndarray, np.ndarray]: The correlated weights of the human and DNN modalities,
    in either the same or different orders based on the parameters.
    """
    dim_human, dim_dnn = weights_human.shape[1], weights_dnn.shape[1]
    weights = {"human": weights_human, "dnn": weights_dnn}
    dims = {"human": dim_human, "dnn": dim_dnn}
    weights_base = weights[base]
    dim_base = dims[base]
    weights_comp = weights["dnn" if base == "human" else "human"]
    dim_comp = dims["dnn" if base == "human" else "human"]

    if dim_base > dim_comp and not duplicates:
        raise ValueError(
            """If duplicates is set to False, the number of dimensions in the base modality
            must be smaller than the number of dimensions in the comparison modality."""
        )

    matching_dims, matching_corrs = [], []
    for i, w1 in enumerate(weights_base.T):
        corrs = np.zeros(dim_comp)
        for j, w2 in enumerate(weights_comp.T):
            corrs[j] = pearsonr(w1, w2)[0]

        sorted_dim_corrs = np.argsort(-corrs)
        if duplicates:
            matching_dims.append(sorted_dim_corrs[0])
        else:
            for dim in sorted_dim_corrs:
                if (
                    dim not in matching_dims
                ):  # take the highest correlation that has not been used before
                    matching_dims.append(dim)
                    break

        # Store the highest correlation for the selected dimension
        select_dim = matching_dims[-1]
        matching_corrs.append(corrs[select_dim])

    # Now sort the dimensions based on the highest correlations
    if sort_by_corrs:
        matching_corrs = np.array(matching_corrs)
        sorted_corrs = np.argsort(-matching_corrs)
        matching_corrs = matching_corrs[sorted_corrs]
        comp_dims = np.array(matching_dims)[sorted_corrs]
        base_dims = sorted_corrs

    else:
        base_dims = np.arange(len(matching_dims))
        comp_dims = np.array(matching_dims)

    weights_base = weights_base[:, base_dims]
    weights_comp = weights_comp[:, comp_dims]

    if return_corrs:
        return weights_base, weights_comp, matching_corrs, matching_dims

    else:
        return weights_base, weights_comp
