import pandas as pd
import torch
from torch import Tensor
from typing import Tuple
from tqdm import tqdm

# from numba import njit, prange
import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr


def load_concepts(path: str = "./data/misc/category_mat_manual.tsv") -> pd.DataFrame:
    """Loads the THINGS concept file into a dataframe"""
    concepts = pd.read_csv(path, encoding="utf-8", sep="\t")
    return concepts


def split_rsm(rsm):
    # Split the RSM into two halves based on the objects
    m = rsm.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)
    half_size = m // 2
    first_half_indices = indices[:half_size]
    second_half_indices = indices[half_size:]

    rsm_1 = np.corrcoef(rsm[first_half_indices, :][:, first_half_indices])
    rsm_2 = np.corrcoef(rsm[second_half_indices, :][:, second_half_indices])

    return rsm_1, rsm_2


def flatten_upper_triangle(matrix):
    # Get the upper triangle indices, excluding the diagonal
    upper_triangle_indices = np.triu_indices_from(matrix, k=1)
    return matrix[upper_triangle_indices]


def split_half_reliability(rsm: np.ndarray):
    rsm_1, rsm_2 = split_rsm(rsm)

    # Flatten the upper triangular parts of the correlation matrices
    rsm_1_flattened = flatten_upper_triangle(rsm_1)
    rsm_2_flattened = flatten_upper_triangle(rsm_2)

    # Compute the Pearson correlation
    split_half_corr, _ = pearsonr(rsm_1_flattened, rsm_2_flattened)

    # Spearman-Brown formula for reliability (i.e., noise ceiling)
    reliability = (2 * split_half_corr) / (1 + split_half_corr)
    return reliability


def fill_diag(rsm):
    """Fill main diagonal of the RSM with ones"""
    assert np.allclose(rsm, rsm.T), "\nRSM is required to be a symmetric matrix\n"
    rsm[np.eye(len(rsm)) == 1.0] = 1

    return rsm


def compute_rdm(X, method="correlation"):
    assert method in ["correlation", "euclidean"]
    if method == "euclidean":
        rdm = squareform(pdist(X, metric="euclidean"))
    else:
        rsm = correlation_matrix(X)
        rdm = 1 - rsm

    return rdm


def compute_rsm(X, method="correlation"):
    assert method in ["correlation", "euclidean"]
    if method == "euclidean":
        rdm = squareform(pdist(X, metric="euclidean"))
        rsm = 1 - rdm
    else:
        rsm = correlation_matrix(X)

    return rsm


def correlation_matrix(F, a_min=-1.0, a_max=1.0):
    # """Compute dissimilarity matrix based on correlation distance (on the matrix-level).
    # Same as np.corrcoef(rowvar=True)"""
    F_c = F - F.mean(axis=1)[..., None]
    cov = F_c @ F_c.T
    l2_norms = np.linalg.norm(F_c, axis=1)  # compute vector l2-norm across rows
    denom = np.outer(l2_norms, l2_norms)

    corr_mat = cov / denom
    corr_mat = np.nan_to_num(corr_mat, nan=0.0)
    corr_mat = corr_mat.clip(min=a_min, max=a_max)
    corr_mat = fill_diag(corr_mat)
    return corr_mat


def correlate_rsms(rsm_a, rsm_b, correlation="pearson", return_pval=False):
    assert correlation in [
        "pearson",
        "spearman",
    ], "Correlation must be pearson or spearman"
    """Correlate the lower triangular parts of two rsms"""
    rsm_a = fill_diag(rsm_a)
    rsm_b = fill_diag(rsm_b)
    triu_inds = np.triu_indices(len(rsm_a), k=1)
    corr_func = (
        scipy.stats.pearsonr if correlation == "pearson" else scipy.stats.spearmanr
    )
    rho, p = corr_func(rsm_a[triu_inds], rsm_b[triu_inds])
    if return_pval:
        return rho, p
    else:
        return rho


def fill_diag_torch(tensor):
    """Set diagonal elements to zero"""
    return tensor - torch.diag(tensor.diag())


def pearson_corr_torch(x, y):
    """Compute Pearson correlation for PyTorch tensors"""
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    return torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2))
    )


def spearman_corr_torch(x, y):
    """Compute Spearman correlation for PyTorch tensors"""
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    return torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2))
    )


def correlate_rsms_torch(rsm_a, rsm_b, correlation="pearson"):
    assert correlation in [
        "pearson",
        "spearman",
    ], "Correlation must be pearson or spearman"
    """Correlate the lower triangular parts of two rsms"""
    rsm_a = fill_diag_torch(rsm_a)
    rsm_b = fill_diag_torch(rsm_b)
    triu_inds = torch.triu_indices(len(rsm_a), len(rsm_a), offset=1)
    corr_func = pearson_corr_torch if correlation == "pearson" else spearman_corr_torch
    rho = corr_func(
        rsm_a[triu_inds[0], triu_inds[1]], rsm_b[triu_inds[0], triu_inds[1]]
    )
    return rho


def rsm_pred_torch(
    embedding: Tensor,
    return_type="numpy",
    verbose=False,
    return_accuracy=False,
) -> Tuple[Tensor, float]:
    """
    Compute the reconstruction similarity matrix (RSM) for a given embedding
    Args:
        W (Tensor): An input tensor representing the embedding,
            with shape (n_objects, embedding_dim), where n_objects is the number
            of objects, and embedding_dim is the dimensionality of the embedding.

    Returns:
        Tuple[Tensor, float]: A tuple containing the RSM as a tensor with shape
            (n_objects, n_objects) and the odd-one-out accuracy as a float.

    Example:
        >>> W = torch.randn(100, 50)
        >>> rsm, odd_one_out_accuracy = rsm_reconstructed(W)
        >>> print(rsm.shape)
        torch.Size([100, 100])
        >>> print(odd_one_out_accuracy)
        0.123456789
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(embedding, torch.Tensor):
        embedding = torch.tensor(embedding, dtype=torch.double, device=device)

    sim_matrix = torch.matmul(embedding, embedding.T)
    sim_matrix = sim_matrix.to(dtype=torch.double, device=device)
    sim_matrix.exp_()
    n_objects = sim_matrix.shape[0]
    indices = torch.triu_indices(n_objects, n_objects, offset=1)

    n_indices = indices.shape[1]
    batch_size = min(n_indices, 10_000)
    n_batches = (n_indices + batch_size - 1) // batch_size

    rsm = torch.zeros_like(sim_matrix).double()
    ooo_accuracy = 0.0
    pbar = tqdm(total=n_batches) if verbose else None

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_indices)
        batch_indices = indices[:, start_idx:end_idx]
        i, j = batch_indices
        s_ij = sim_matrix[i, j]
        s_ik = sim_matrix[i, :]
        s_jk = sim_matrix[j, :]

        # This is a vectorized definition of 'for k in range(n_objects): if k != i and k != j'
        # By setting this to 0, we can vectorize the for loop by seeting the softmax
        # to 1 if i==k or j==k and then subtract 2 from the sum of the softmax.
        n = end_idx - start_idx
        n_range = np.arange(n)
        s_ik[n_range, i] = 0
        s_ik[n_range, j] = 0
        s_jk[n_range, j] = 0
        s_jk[n_range, i] = 0

        s_ij = s_ij.unsqueeze(1)
        softmax_ij = s_ij / (s_ij + s_jk + s_ik)
        proba_sum = softmax_ij.sum(1) - 2
        mean_proba = proba_sum / (n_objects - 2)
        rsm[i, j] = mean_proba
        ooo_accuracy += mean_proba.mean()

        if verbose:
            pbar.set_description(f"Batch {batch_idx+1}/{n_batches}")
            pbar.update(1)

    pbar.close() if verbose else None
    rsm = rsm.cpu().numpy()
    rsm += rsm.T  # make similarity matrix symmetric

    np.fill_diagonal(rsm, 1)
    ooo_accuracy = ooo_accuracy.item() / n_batches

    if return_type == "tensor":
        if return_accuracy:
            return torch.tensor(rsm, dtype=torch.float32), ooo_accuracy
        return torch.tensor(rsm, dtype=torch.float32)
    else:
        if return_accuracy:
            return rsm, ooo_accuracy
        return rsm


def rsm_pred_numpy(W: np.ndarray) -> np.ndarray:
    """convert weight matrix corresponding to the mean of each dim distribution for an object into a RSM"""
    N = W.shape[0]
    S = np.matmul(W, W.T)
    S_e = np.exp(S)  # exponentia
    rsm = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(N):
                if k != i and k != j:
                    rsm[i, j] += S_e[i, j] / (S_e[i, j] + S_e[i, k] + S_e[j, k])
    rsm /= N - 2
    rsm += rsm.T  # make similarity matrix symmetric
    np.fill_diagonal(rsm, 1)
    return rsm


# @njit(parallel=False, fastmath=False)
# def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
#     I, K = A.shape
#     K, J = B.shape
#     C = np.zeros((I, J))
#     for i in prange(I):
#         for j in prange(J):
#             for k in prange(K):
#                 C[i, j] += A[i, k] * B[k, j]
#     return C


# @njit(parallel=False, fastmath=False)
# def rsm_pred_numba(W: np.ndarray) -> np.ndarray:
#     """convert weight matrix corresponding to the mean of each dim distribution for an object into a RSM"""
#     N = W.shape[0]
#     S = matmul(W, W.T)
#     S_e = np.exp(S)  # exponentiate all elements in the inner product matrix S
#     rsm = np.zeros((N, N))
#     ooo_acc = 0.0
#     for i in prange(N):
#         for j in prange(i + 1, N):
#             for k in prange(N):
#                 if k != i and k != j:
#                     rsm[i, j] += S_e[i, j] / (S_e[i, j] + S_e[i, k] + S_e[j, k])

#     rsm /= N - 2
#     rsm += rsm.T  # make similarity matrix symmetric
#     np.fill_diagonal(rsm, 1)
#     return rsm
