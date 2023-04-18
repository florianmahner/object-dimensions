import torch
import numpy as np
from object_dimensions import ExperimentParser
from object_dimensions.utils.utils import (
    load_sparse_codes,
    load_image_data,
    correlate_rsms,
)
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


def rsm_reconstructed(embedding: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute the reconstruction similarity matrix for a given embedding."""
    sim_matrix = embedding @ embedding.T
    sim_matrix = torch.tensor(sim_matrix).double()
    sim_matrix = sim_matrix.exp().cuda()

    n_objects = len(sim_matrix)
    indices = torch.tril_indices(n_objects, n_objects, offset=-1)
    batch_size = 10_000
    n_indices = indices.shape[1]
    n_batches = (indices.shape[1] + batch_size - 1) // batch_size
    n_batches = min(1000, n_batches)

    rsm = torch.zeros_like(sim_matrix).double()
    ooo_accuracy = 0.0
    pbar = tqdm(total=n_batches, miniters=1)

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_indices)

        batch_indices = indices[:, start_idx:end_idx]
        i, j = batch_indices[0], batch_indices[1]

        s_ij = sim_matrix[i, j]
        s_jk = sim_matrix[:, j]
        s_ik = sim_matrix[i, :]

        # I want at each column to set the value of s_ik to zero at index i
        # and the value of s_jk to zero at index j. Using this the softmax
        # will be 1 for the i and j indices.
        n = end_idx - start_idx
        n_range = np.arange(n)
        s_ik[n_range, i] = 0
        s_jk[j, n_range] = 0
        softmax_ij = s_ij / (s_ij + s_jk + s_ik.T)

        # We take the mean over all k, but we need to exclude the i and j that is
        # why we subtract 2 from each column
        proba_sum = softmax_ij.sum(0) - 2
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
    rsm_dnn, ooo_dnn = rsm_reconstructed(dnn_embedding)
    rsm_human, ooo_human = rsm_reconstructed(human_embedding)
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
