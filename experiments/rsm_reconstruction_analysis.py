import time
import torch
from typing import Tuple
import torch
from torch import Tensor
from tqdm import tqdm
from numba import njit, prange
import numpy as np


def rsm_pred_torch(embedding: Tensor) -> Tuple[Tensor, float]:
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
    pbar = tqdm(total=n_batches)

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

        pbar.set_description(f"Batch {batch_idx+1}/{n_batches}")
        pbar.update(1)

    pbar.close()
    rsm = rsm.cpu().numpy()
    rsm += rsm.T  # make similarity matrix symmetric

    np.fill_diagonal(rsm, 1)
    ooo_accuracy = ooo_accuracy.item() / n_batches
    return rsm, ooo_accuracy


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


@njit(parallel=False, fastmath=False)
def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    I, K = A.shape
    K, J = B.shape
    C = np.zeros((I, J))
    for i in prange(I):
        for j in prange(J):
            for k in prange(K):
                C[i, j] += A[i, k] * B[k, j]
    return C


@njit(parallel=False, fastmath=False)
def rsm_pred_numba(W: np.ndarray) -> np.ndarray:
    """convert weight matrix corresponding to the mean of each dim distribution for an object into a RSM"""
    N = W.shape[0]
    S = matmul(W, W.T)
    S_e = np.exp(S)  # exponentiate all elements in the inner product matrix S
    rsm = np.zeros((N, N))
    ooo_acc = 0.0
    for i in prange(N):
        for j in prange(i + 1, N):
            for k in prange(N):
                if k != i and k != j:
                    rsm[i, j] += S_e[i, j] / (S_e[i, j] + S_e[i, k] + S_e[j, k])

    rsm /= N - 2
    rsm += rsm.T  # make similarity matrix symmetric
    np.fill_diagonal(rsm, 1)
    return rsm


def check_time(func, *args, **kwargs):
    """Check the time taken by a function to run"""
    start = time.time()
    out = func(*args, **kwargs)
    end = time.time()
    print(f"{func.__name__} took {end-start:.2f} seconds to run")
    return out[0] if isinstance(out, tuple) else out


if "__main__" == __name__:
    # Create a random embedding
    np.random.seed(42)
    W_np = np.random.randn(400, 100)
    W_torch = torch.tensor(W_np, dtype=torch.float64)

    # Calculate RSMs using all three functions
    rsm_torch = check_time(rsm_pred_torch, W_torch)
    rsm_numpy = check_time(rsm_pred_numpy, W_np)
    rsm_numba = check_time(rsm_pred_numba, W_np)

    # Set the tolerance value for comparing RSMs
    rtol = 1e-12
    atol = 1e-12

    # print if rsm torch and rsm numpy are equal
    print(
        "RSM torch and RSM numpy are equal:",
        np.allclose(rsm_torch, rsm_numpy, rtol=rtol, atol=atol),
    )

    # print if rsm numpy and rsm numba are equal
    print(
        "RSM numpy and RSM numba are equal:",
        np.allclose(rsm_numpy, rsm_numba, rtol=rtol, atol=atol),
    )

    print(
        "RSM torch and RSM numba are equal:",
        np.allclose(rsm_torch, rsm_numba, rtol=rtol, atol=atol),
    )
