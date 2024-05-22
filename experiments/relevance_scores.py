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
from scipy.stats import pearsonr


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

# dimension_mapping = pd.DataFrame(
#     {
#         "Human Dimension": dimension_mapping["Human Dimension"],
#         "match_unique": dimension_mapping["match_unique"],
#         "match_duplicates": dimension_mapping["match_duplicates"],
#     }
# )
dimension_ratings = pd.read_csv(
    "/LOCAL/fmahner/object-dimensions/data/misc/dimension_ratings_processed.csv"
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
        # softmax_diff[:, :, i] = np.maximum(softmax_default - softmax_per_batch, 0)
        softmax_diff[:, :, i] = softmax_default - softmax_per_batch

    softmax_diff = softmax_diff + 1e-12

    return softmax_default, softmax_diff


# %%

# these are the validation dataset triplets
# triplets = val_loader.dataset[:]


# create a random sample of 5 million triplets of shape n, 3
n = 1_000_000
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

# Here we now sort the dimension based on the dimension correlations we obtained previously
unique = dimension_mapping["match_unique"]
duplicates = dimension_mapping["match_duplicates"]

# Do pairwise matching of human and dnn dimensions
for i, row in dimension_mapping.iterrows():
    # relevance_dnn[:, :, i] = relevance_dnn[:, :, row["match_unique"]]
    relevance_dnn[:, i] = relevance_dnn[:, row["match_unique"]]

# Drop the last two dimensions of the DNN
relevance_dnn_unique = relevance_dnn[:, :-2]
# relevance_dnn_unique = relevance_dnn[:, :, :-2]


# %%

# Find the relevances for human and dnn where both have the same argmax
argmax_human = np.argmax(softmax_dnn, 1)
argmax_dnn = np.argmax(softmax_human, 1)

# argmax_human = np.argmax(relevance_human, 1)
# argmax_dnn = np.argmax(relevance_dnn_unique, 1)


# i want to find the intersection based on the m dimension
intersection = np.where(argmax_human == argmax_dnn)[0]
diff = np.where(argmax_human != argmax_dnn)[0]


# %%


def vectorized_corrcoef(matrix1, matrix2):
    """Compute correlation coefficients between corresponding rows of two matrices"""
    assert matrix1.shape == matrix2.shape, "Matrices must have the same shape."
    mean1 = matrix1.mean(axis=1, keepdims=True)
    mean2 = matrix2.mean(axis=1, keepdims=True)
    centered1 = matrix1 - mean1
    centered2 = matrix2 - mean2

    # Compute the numerator (covariance)
    numerator = np.sum(centered1 * centered2, axis=1)

    # Compute the denominator (product of standard deviations)
    std1 = np.sqrt(np.sum(centered1**2, axis=1))
    std2 = np.sqrt(np.sum(centered2**2, axis=1))
    denominator = std1 * std2

    # Compute Pearson correlation coefficients
    correlations = numerator / denominator
    return correlations


def cosine_similarity(matrix1, matrix2):
    """Compute cosine similairty between corresponding rows of two matrices"""
    assert matrix1.shape == matrix2.shape, "Matrices must have the same shape."
    dot_product = np.sum(matrix1 * matrix2, axis=1)
    norm1 = np.sqrt(np.sum(matrix1**2, axis=1))
    norm2 = np.sqrt(np.sum(matrix2**2, axis=1))
    similarity = dot_product / (norm1 * norm2)
    return similarity


# Extract relevant rows
relevance_human_intersection = relevance_human[intersection]
relevance_dnn_intersection = relevance_dnn_unique[intersection]


relevance_human_intersection = relevance_human_intersection.reshape(
    len(intersection), -1
)
relevance_dnn_intersection = relevance_dnn_intersection.reshape(len(intersection), -1)

relevance_human_diff = relevance_human[diff]
relevance_dnn_diff = relevance_dnn_unique[diff]

relevance_human_diff = relevance_human_diff.reshape(len(diff), -1)
relevance_dnn_diff = relevance_dnn_diff.reshape(len(diff), -1)

corrs_intersection = vectorized_corrcoef(
    relevance_human_intersection, relevance_dnn_intersection
)
corrs_diff = vectorized_corrcoef(relevance_human_diff, relevance_dnn_diff)

# corrs_intersection = cosine_similarity(
#     relevance_human_intersection, relevance_dnn_intersection
# )
# corrs_diff = cosine_similarity(relevance_human_diff, relevance_dnn_diff)


# %%

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


# %%

# THIS is the pie chart plotting!

# find the top relevant dimensions
relevance_ooo_human = relevance_human[:, 0, :]

# NOTE I dont take the unique dimensions here
relevance_ooo_dnn = relevance_dnn[:, 0, :]

# find the top relevant dimensions
topk = 1
top_relevance_human = np.argsort(relevance_ooo_human, axis=1)[:, -topk:]
top_relevance_dnn = np.argsort(relevance_ooo_dnn, axis=1)[:, -topk:]

ratings_human = dimension_ratings[dimension_ratings["Model"] == "Human"].reset_index()
ratings_dnn = dimension_ratings[dimension_ratings["Model"] == "VGG16"].reset_index()


quality_human = ratings_human.loc[top_relevance_human.flatten()]["Quality"].values
quality_dnn = ratings_dnn.loc[top_relevance_dnn.flatten()]["Quality"].values


# Create a mapping from quality labels to integers
unique_labels = np.unique(dimension_ratings["Quality"])
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

# Convert quality labels to integer values
quality_human_int = np.array([label_to_int[label] for label in quality_human])
quality_dnn_int = np.array([label_to_int[label] for label in quality_dnn])

# Make a pie chart of the quality of the top relevant dimensions
sns.set_palette("deep")
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].pie(np.bincount(quality_human_int), labels=unique_labels, autopct="%1.1f%%")
ax[0].set_title("Human")
ax[1].pie(np.bincount(quality_dnn_int), labels=unique_labels, autopct="%1.1f%%")
ax[1].set_title("VGG16")
fig.savefig(
    "/LOCAL/fmahner/object-dimensions/results/plots/top_relevance_qualities_human_dnn.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
