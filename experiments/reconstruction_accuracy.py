import time
import torch
import numpy as np
import random
from object_dimensions import ExperimentParser
from object_dimensions.utils.utils import load_sparse_codes
from scipy.stats import pearsonr

parser = ExperimentParser()
parser.add_argument("--human_embedding", type=str, default="./results/human")
parser.add_argument("--dnn_embedding", type=str, default="./results/dnn")


def compute_argmax_accuracy(similarity_matrix, device):
    """Compute the accuracy of the argmax of the similarity matrix. The accuracy is
    computed by counting the number of times the argmax of the similarity matrix is
    (i,j) for a random (i,j) tuple and all (i,k) and (k,j) tuples.)"""
    n_objects = similarity_matrix.shape[0]
    similarity_matrix = torch.from_numpy(similarity_matrix).to(device)

    # Set the diagonal to zero
    similarity_matrix = similarity_matrix - torch.diag(torch.diag(similarity_matrix))

    # Create tuples for only the lower triangular part, offset by -1 to exclude the diagonal
    indices = torch.tril_indices(n_objects, n_objects, offset=-1).T
    indices = indices.to(device)

    # Select a random subset of the indices
    random.seed(0)
    indices = indices[random.sample(range(len(indices)), 1_000_000)]

    n_ij = len(indices)
    cell_accuracy = torch.zeros(n_ij, device=device)

    start_time = time.time()
    for ctr in range(n_ij):
        i, j = indices[ctr]

        s_ij = similarity_matrix[i, j]
        s_kj = similarity_matrix[:, j]
        s_ik = similarity_matrix[i, :]
        values = torch.stack([s_ij.repeat(n_objects), s_kj, s_ik], dim=1)

        # Use torch.argmax to get the indices of the max values along the first dimension.
        argmax_indices = torch.argmax(values, dim=1)

        # Count the number of times the argmax is (i,j)
        # This give me the probability of the argmax being (i,j) for a random (i,j) tuple given all k
        argmax_ij = (argmax_indices == 0).sum()

        # Compute the accuracy by dividing by the number of objects (k)
        argmax_accuracy = argmax_ij / n_objects
        cell_accuracy[ctr] = argmax_accuracy

        print("Ctr: ", ctr, end="\r")

        # Print time used for 1 million iterations
        if ctr + 1 % 1_000_000 == 0:
            print(f"Time used for 1 million iterations: {time.time() - start_time}")
            start_time = time.time()

    cell_accuracy = cell_accuracy.cpu().numpy()
    return cell_accuracy


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    human_embedding = load_sparse_codes(args.human_embedding)
    dnn_embedding = load_sparse_codes(args.dnn_embedding)
    sim_human = human_embedding @ human_embedding.T
    sim_dnn = dnn_embedding @ dnn_embedding.T

    accuracy_dnn = compute_argmax_accuracy(sim_dnn, device)
    accuracy_human = compute_argmax_accuracy(sim_human, device)

    print("DNN accuracy: ", accuracy_dnn.mean())
    print("Human accuracy: ", accuracy_human.mean())

    breakpoint()

    corr = pearsonr(accuracy_dnn, accuracy_human)[0]
    print("Correlation: ", corr)


if __name__ == "__main__":
    args = parser.parse_args()
    args.human_embedding = "./data/misc/vice_embedding_lukas_66d.txt"
    args.dnn_embedding = "results/sslab_final/deep/vgg16_bn/classifier.3/20.mio/sslab/300/256/0.24/0/params/parameters.npz"
    main(args)
