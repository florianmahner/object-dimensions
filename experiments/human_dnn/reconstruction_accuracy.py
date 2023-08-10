import torch
import numpy as np
from tomlparse import argparse
from object_dimensions.utils.utils import (
    load_sparse_codes,
    load_image_data,
    correlate_rsms,
)
from torch import Tensor
from tqdm import tqdm
from typing import Tuple, Union


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_path", type=str, default="./results/human")
    parser.add_argument("--dnn_path", type=str, default="./results/dnn")
    parser.add_argument(
        "--img_root",
        type=str,
        default="./data/reference_images",
        help="Path to image root directory",
    )
    return parser.parse_args()


def rsm_pred_torch(
    embedding: Union[np.ndarray, torch.Tensor],
    w_acc: bool = False,
    verbose: bool = False,
    return_type: str = "numpy",
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
    assert isinstance(
        embedding, (np.ndarray, torch.Tensor)
    ), "embedding must be a numpy array or torch tensor"
    if isinstance(embedding, np.ndarray):
        embedding = torch.tensor(embedding, dtype=torch.double)
    if embedding.dtype != torch.double:
        embedding = embedding.double()

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

    if verbose:
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

        if verbose:
            pbar.set_description(f"Batch {batch_idx+1}/{n_batches}")
            pbar.update(1)

    if verbose:
        pbar.close()

    rsm = rsm + rsm.T  # make similarity matrix symmetric
    rsm.fill_diagonal_(1)
    ooo_accuracy = ooo_accuracy.item() / n_batches

    if return_type == "numpy":
        rsm = rsm.cpu().numpy()

    if w_acc:
        return rsm, ooo_accuracy
    else:
        return rsm


def main(args):
    human_embedding = load_sparse_codes(args.human_path)
    dnn_embedding = load_sparse_codes(args.dnn_path)
    indices = load_image_data(args.img_root, filter_behavior=True)[1]
    dnn_embedding = dnn_embedding[indices]
    rsm_dnn, ooo_dnn = rsm_pred_torch(dnn_embedding, w_acc=True, verbose=True)
    rsm_human, ooo_human = rsm_pred_torch(human_embedding, w_acc=True, verbose=True)
    rho = correlate_rsms(rsm_dnn, rsm_human)
    print(
        f"RSM correlation: {rho:.9f}, OOO DNN: {ooo_dnn:.5f}, OOO human: {ooo_human:.5f}",
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
