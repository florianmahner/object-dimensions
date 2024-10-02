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


# %%

# %%


import seaborn as sns

# Calculate average similarity for each dimension
n_dimensions = dnn.shape[1]
top_objects = 50

for dim in range(n_dimensions):
    dim_values = model[:, dim]

    # Take top 2
    top_indices = np.argsort(-dim_values)[:top_objects]
    top_sensevec = sensevec[top_indices]

    # Get the similarity of the top sensevec vectors
    corrs = np.corrcoef(top_sensevec)

    # Calculate mean similarity of upper triangular (excluding diagonal)
    upper_triangular = np.triu(corrs, k=1)
    mean_similarity = np.mean(upper_triangular)
    df_similarities.loc[dim, "Average_Similarity"] = mean_similarity

    bottom_indices = np.argsort(dim_values)[:top_objects]
    bottom_sensevec = sensevec[bottom_indices]

    bottom_similarities = []
    for i, top in enumerate(top_sensevec):
        bottom_similarity = np.corrcoef(top, bottom_sensevec[i])
        bottom_similarities.append(np.average(np.triu(bottom_similarity, k=1)))

    mean_bottom_similarity = np.mean(bottom_similarities)
    df_similarities.loc[dim, "Bottom_Similarities"] = mean_bottom_similarity

    df_similarities.loc[dim, "Botto_Corrected_Similarities"] = (
        mean_similarity - mean_bottom_similarity
    )

    # Assign type to dimension
    if dim in semantic_dimensions:
        df_similarities.loc[dim, "Type"] = "Semantic"
    elif dim in visual_dimensions:
        df_similarities.loc[dim, "Type"] = "Visual"
    elif dim in mixed_dimensions:
        df_similarities.loc[dim, "Type"] = "Mixed Visual-Semantic"
    elif dim in unclear_dimensions:
        df_similarities.loc[dim, "Type"] = "Unclear"
    else:
        df_similarities.loc[dim, "Type"] = "Unknown"
# Plot the average similarities as a bar plot using seaborn
plt.figure(figsize=(12, 6))
sns.barplot(
    x="Dimension",
    y="Botto_Corrected_Similarities",
    hue="Type",
    data=df_similarities,
    palette={
        "Visual": "blue",
        "Semantic": "red",
        "Mixed Visual-Semantic": "green",
        "Unclear": "grey",
    },
)

plt.xlabel("Dimension")
plt.ylabel("Average SenseVec Similarity")
plt.legend(title="Human Label", loc="upper right")

# Set x-axis ticks to every 10 dimensions
plt.xticks(range(0, n_dimensions, 10))

plt.tight_layout()
plt.show()

# Compare average similarities
semantic_similarities = df_similarities[df_similarities["Type"] == "Semantic"][
    "Average_Similarity"
]
visual_similarities = df_similarities[df_similarities["Type"] == "Visual"][
    "Average_Similarity"
]

print(f"Average similarity for semantic dimensions: {semantic_similarities.mean():.4f}")
print(f"Average similarity for visual dimensions: {visual_similarities.mean():.4f}")


# %%

# %%

# Mittlere ähnlichkeit der ersten 20.
# nimm erstes objekte und vergleiche mit den letzten 19.
# von jeder der 20 schaue ich mir die ähnlichkeit zu den letzten 20.

# erst mal nur die baseline plotten.
# baseline nochmal abziehen

# %%

""""
zweite idee.
lineares modell, dimensionen von vorhersagen aus semantischem embedding.
lineare funktion draufgefitted, wie kann man die sparsity. 

okay, dann probiere erstmal.

does it show the effect of visual vs semantic dimensions.
"""
