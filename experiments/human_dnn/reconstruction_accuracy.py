import torch
import numpy as np
from object_dimensions import ExperimentParser
from object_dimensions.utils.utils import (
    load_sparse_codes,
    load_image_data,
    correlate_rsms,
)
from torch import Tensor
from scipy.stats import pearsonr
from tqdm import tqdm
from typing import Tuple


parser = ExperimentParser()
parser.add_argument("--human_embedding", type=str, default="./results/human")
parser.add_argument("--dnn_embedding", type=str, default="./results/dnn")
parser.add_argument(
    "--img_root",
    type=str,
    default="./data/reference_images",
    help="Path to image root directory",
)


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


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    human_embedding = load_sparse_codes(args.human_embedding)
    dnn_embedding = load_sparse_codes(args.dnn_embedding)
    indices = load_image_data(args.img_root, filter_behavior=True)[1]
    dnn_embedding = dnn_embedding[indices]
    dnn_embedding = torch.tensor(dnn_embedding)
    human_embedding = torch.tensor(human_embedding)
    rsm_dnn, ooo_dnn = rsm_pred_torch(dnn_embedding)
    rsm_human, ooo_human = rsm_pred_torch(human_embedding)
    rho = correlate_rsms(rsm_dnn, rsm_human)
    print(
        f"RSM correlation: {rho:.9f}, OOO DNN: {ooo_dnn:.5f}, OOO human: {ooo_human:.5f}",
    )


if __name__ == "__main__":
    args = parser.parse_args()
    args.human_embedding = "./data/misc/vice_embedding_lukas_66d.txt"
    args.img_root = "./data/image_data/images12_plus"
    args.dnn_embedding = "results/sslab_final/deep/vgg16_bn/classifier.3/20.mio/sslab/300/256/0.24/0/params/parameters.npz"
    main(args)
