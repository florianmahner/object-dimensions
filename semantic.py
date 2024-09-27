# %%
import pandas as pd
import numpy as np
from scipy.io import loadmat
from objdim.utils import load_sparse_codes, load_image_data
import matplotlib.pyplot as plt

from experiments.human_labeling.dnn_dimension_ratings import (
    load_dimension_mapping,
    process_dimension_ratings,
)


# %%

# Load the dnn embedding

dnn = load_sparse_codes("./data/embeddings/vgg16_bn/classifier.3/")
images, indices = load_image_data("./data/images/things", filter_behavior=True)
dnn = dnn[indices, :]


dimension_mapping_path = "./data/misc/dimension_mapping.json"
dimension_ratings_path = "./data/misc/dimension_ratings.mat"
dimension_mapping = load_dimension_mapping(dimension_mapping_path)
ratings = loadmat(dimension_ratings_path)["ratings"]
df = process_dimension_ratings(ratings, dimension_mapping)


# %%


A = loadmat("/LOCAL/fmahner/object-dimensions/data/misc/sense-wordvec.mat")
A = A["sensevec_augmented"]

# Delete objects with nan rows
nan_rows = np.isnan(A).any(axis=1)
A = A[~nan_rows]

B = dnn[~nan_rows]

# %%

# Parameters
k = 10  # Number of top objects to consider

# Initialize list to store results
visual_semantic_distribution = []

# Iterate over each DNN dimension
for i in range(B.shape[1]):
    # Get the top k objects for the current DNN dimension
    top_k_indices = np.argsort(B[:, i])[-k:]
    top_k_dnn = B[top_k_indices, i]
    top_k_sensevec = A[top_k_indices, :]

    # Compute correlation with each sensevec dimension
    correlations = np.corrcoef(top_k_dnn, top_k_sensevec.T)[0, 1:]

    # Normalize correlations to form a probability distribution
    correlations = np.abs(correlations)  # Take absolute value of correlations
    prob_distribution = correlations / np.sum(correlations)

    # Store the distribution
    visual_semantic_distribution.append(prob_distribution)

# Print results
for i, distribution in enumerate(visual_semantic_distribution):
    print(f"DNN Dimension {i}: {distribution}")


# %%

# Parameters
k = 10  # Number of top objects to consider

# Initialize list to store results
visual_or_semantic = []

# Iterate over each DNN dimension
for i in range(B.shape[1]):
    # Get the top k objects for the current DNN dimension
    top_k_indices = np.argsort(B[:, i])[-k:]
    top_k_dnn = B[top_k_indices, i]
    top_k_sensevec = A[top_k_indices, :]

    # Compute correlation with each sensevec dimension
    correlations = np.corrcoef(top_k_dnn, top_k_sensevec.T)[0, 1:]

    # Determine if the dimension is visual or semantic based on the gradient
    gradient = np.gradient(correlations)
    if np.mean(gradient) > 0:  # Define a threshold for gradient
        visual_or_semantic.append("semantic")
    else:
        visual_or_semantic.append("visual")

# %%
# make one vertical bar plot of the number of visual and semantic dimensions
plt.bar(visual_or_semantic, np.ones(len(visual_or_semantic)))
plt.show()
