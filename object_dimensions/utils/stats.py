import torch
import math
import numpy as np
from typing import Union


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
