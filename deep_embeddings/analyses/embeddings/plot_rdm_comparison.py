import numpy as np
import matplotlib.pyplot as plt
from deep_embeddings.utils import compute_rdm

""" Compute RDMs for two embeddings and plot them. We have not clustered based on the category / class names yet. This would give
an even nicer looking embedding! """

embedding_1 = "../learned_embeddings/spose_embedding_66d_sorted.txt"
embedding_2 = "../learned_embeddings/weights_things_behavior_8196bs_adaptive_half/params/pruned_q_mu_epoch_5000.txt"

embedding_1 = np.loadtxt(embedding_1)
embedding_1 = np.maximum(0, embedding_1)
embedding_2 = np.loadtxt(embedding_2)
embedding_2 = np.maximum(0, embedding_2)

method = "correlation"

rsm_1 = compute_rdm(embedding_1, method)
rsm_2 = compute_rdm(embedding_1, method)


# Perform k-means clustering on the rsm matrices -> has not yet worked!
# kmeans = KMeans(n_clusters=30, random_state=0).fit(rsm_1)
# rsm_1 = kmeans.transform(rsm_1)
# rsm_2 = kmeans.transform(rsm_2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(rsm_1, cmap="BuPu")
ax1.set_title("Behavior")
ax2.imshow(rsm_2, cmap="BuPu")
ax2.set_title("VICE")

plt.tight_layout()
fig.savefig("../plots/rdm_comparison_spose_vice_triplets_behavior_8196bs.png", dpi=100)