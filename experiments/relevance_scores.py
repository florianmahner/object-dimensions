# %%

from experiments.human_dnn.jackknife import (
    build_dataloader,
    compute_softmax_per_batch,
)
from object_dimensions.utils import (
    load_sparse_codes,
    load_image_data,
)
import numpy as np

import seaborn as sns
import torch
import matplotlib.pyplot as plt


# %%
# If all data on GPU, num workers need to be 0 and pin memory false
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

img_root = "/LOCAL/fmahner/object-dimensions/data/image_data/images12_plus"
human_path = "/LOCAL/fmahner/object-dimensions/results/behavior/variational/4.12mio/sslab/150/256/1.0/21/params/parameters.npz"
dnn_path = "/LOCAL/fmahner/object-dimensions/results/dnn/variational/vgg16_bn/classifier.3/20.mio/sslab/150/256/1.0/14/params/parameters.npz"

dnn_weights, dnn_var = load_sparse_codes(dnn_path, with_var=True, relu=True)
human_weights, human_var = load_sparse_codes(human_path, with_var=True, relu=True)

triplet_path = "/LOCAL/fmahner/object-dimensions/data/triplets/behavior"
val_loader = build_dataloader(triplet_path)


# Load image data
image_filenames, indices = load_image_data(img_root, filter_behavior=True)
dnn_weights = dnn_weights[indices]
dnn_var = dnn_var[indices]


# dimension_mapping
import pandas as pd

dimension_mapping = pd.read_csv(
    "/LOCAL/fmahner/object-dimensions/results/dnn/variational/vgg16_bn/classifier.3/20.mio/sslab/150/256/1.0/14/analyses/human_dnn/matching_dims.csv"
)


# %%
def jackknife(q_mu, q_var, triplet_indices, device):
    """This function computes the jackknife analysis for a given embedding"""
    q_mu = torch.tensor(q_mu).to(device)
    q_var = torch.tensor(q_var).to(device)

    n_dims = q_mu.shape[1]
    softmax_diff = np.ones((len(triplet_indices), 3, n_dims)) * float("inf")

    # Without jackknifing for that triplet
    softmax_default = compute_softmax_per_batch(q_mu, q_var, triplet_indices, device)
    softmax_default = softmax_default.detach().cpu().numpy()

    for i in range(n_dims):
        # Take all elements except the i-th embedding dimension
        q_mu_i = torch.cat([q_mu[:, 0:i], q_mu[:, i + 1 :]], dim=1)
        q_var_i = torch.cat([q_var[:, 0:i], q_var[:, i + 1 :]], dim=1)

        # Compute the softmax decisions
        softmax_per_batch = compute_softmax_per_batch(
            q_mu_i, q_var_i, triplet_indices, device
        )

        # This is the odd one out probability (at the index of the odd one out)
        softmax_per_batch = softmax_per_batch.detach().cpu().numpy()
        softmax_diff[:, :, i] = np.abs(softmax_per_batch - softmax_default)
        # softmax_diff[:, :, i] = softmax_per_batch

    return softmax_default, softmax_diff


# %%

# these are the validation dataset triplets
# triplets = val_loader.dataset[:]


# create a random sample of 5 million triplets of shape n, 3
n = 10_000_000
m = 1854
triplets = np.random.randint(0, m, (n, 3))
triplets = torch.tensor(triplets)

# %%
softmax_human, relevance_human = jackknife(human_weights, human_var, triplets, device)

softmax_dnn, relevance_dnn = jackknife(dnn_weights, dnn_var, triplets, device)

# i only look into the relevance of selecting the odd one out, ie the softmax of ij
relevance_human = relevance_human[:, 0, :]
relevance_dnn = relevance_dnn[:, 0, :]


# %%

# Here we now sort the dimension based on the dimension correlations

unique = dimension_mapping["match_unique"]
duplicates = dimension_mapping["match_duplicates"]


dimensions = list(range(70))
# find the dimension that is not in unique
unique = list(unique)
duplicates = list(duplicates)
for dim in unique:
    if dim in dimensions:
        dimensions.remove(dim)

# remove the relevance scores for the dnn not in placde
relevance_dnn_unique = np.delete(relevance_dnn.copy(), dimensions, axis=1)

# TODO do duplicates later


# %%

# Find the relevances for human and dnn where both have the same argmax

argmax_human = np.argmax(softmax_dnn, 1)
argmax_dnn = np.argmax(softmax_human, 1)

argmax_human = np.argmax(relevance_human, 1)
argmax_dnn = np.argmax(relevance_dnn_unique, 1)


# relevance is of shaoe m, n, p. i want to find the argmax of the second dimension
# argmax_human = np.argmax(relevance_human, 1)
# argmax_dnn = np.argmax(relevance_dnn_unique, 1)


# i want to find the intersection based on the m dimension
intersection = np.where(argmax_human == argmax_dnn)[0]
diff = np.where(argmax_human != argmax_dnn)[0]


# %%


def vectorized_corrcoef(matrix1, matrix2):
    """Compute correlation coefficients between corresponding rows of two matrices"""
    mean1 = np.mean(matrix1, axis=1, keepdims=True)
    mean2 = np.mean(matrix2, axis=1, keepdims=True)
    std1 = np.std(matrix1, axis=1, ddof=1, keepdims=True)
    std2 = np.std(matrix2, axis=1, ddof=1, keepdims=True)

    cov = np.mean((matrix1 - mean1) * (matrix2 - mean2), axis=1)
    corr = cov / (std1.flatten() * std2.flatten())

    return corr


# Extract relevant rows
relevance_human_intersection = relevance_human[intersection]
relevance_dnn_intersection = relevance_dnn_unique[intersection]

relevance_human_diff = relevance_human[diff]
relevance_dnn_diff = relevance_dnn_unique[diff]

# Compute correlations for intersection and diff using vectorized function
corrs_intersection = vectorized_corrcoef(
    relevance_human_intersection, relevance_dnn_intersection
)
corrs_diff = vectorized_corrcoef(relevance_human_diff, relevance_dnn_diff)

# If you prefer the results as lists
corrs_intersection = corrs_intersection.tolist()
corrs_diff = corrs_diff.tolist()

# %%


fig, ax = plt.subplots()
sns.histplot(
    corrs_intersection,
    label="same choice",
    ax=ax,
    stat="probability",
    kde=True,
    bins=300,
)
sns.histplot(
    corrs_diff,
    label="different choice",
    ax=ax,
    stat="probability",
    kde=True,
    bins=300,
)
sns.despine(offset=10)
ax.set_xlabel("Pearson correlation")
ax.set_ylabel("Normalized Count")
ax.legend()
fig.savefig(
    "correlation_relevance_scores.png", dpi=300, bbox_inches="tight", pad_inches=0.05
)


# %%


mean_relevance = np.mean(relevance_human_intersection, axis=0)


# %%
