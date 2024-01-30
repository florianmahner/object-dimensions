#%%

import numpy as np
import matplotlib.pyplot as plt
import cv2
from object_dimensions.utils.utils import load_image_data
from scipy.stats import rankdata
from scipy.spatial.distance import pdist, squareform


def kmeans(mat, K=50):
    """write a function that does k means clustering on an image"""
    # convert to np.float32
    Z = mat.reshape(-1, 1)
    Z = np.float32(Z)

    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((mat.shape))
    return res2


def kmeans_sklearn(img, k):
    """Write a function that segments a 2d grayscale image using kmeans clustering and sklearn"""
    from sklearn.cluster import KMeans

    img_flat = img.reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_flat)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    img_kmeans = centers[labels].reshape(img.shape)
    return img_kmeans


def get_rdm(embedding, method="correlation"):
    embedding = np.maximum(0, embedding)
    rsm = compute_rdm(embedding, method)
    rsm = fill_diag(rsm)
    rsm = rankdata(rsm).reshape(rsm.shape)

    return rsm


def normalise_rdm(rdm):
    rdm = rdm / np.max(rdm)
    return rdm


def fill_diag(rsm):
    """Fill main diagonal of the RSM with ones"""
    assert np.allclose(rsm, rsm.T), "\nRSM is required to be a symmetric matrix\n"
    rsm[np.eye(len(rsm)) == 1.0] = 1

    return rsm


def correlation_matrix(F, a_min=-1.0, a_max=1.0):
    """Compute dissimilarity matrix based on correlation distance (on the matrix-level)."""
    F_c = F - F.mean(axis=1)[:, np.newaxis]
    cov = F_c @ F_c.T
    # compute vector l2-norm across rows
    l2_norms = np.linalg.norm(F_c, axis=1)
    denom = np.outer(l2_norms, l2_norms)

    corr_mat = (cov / denom).clip(min=a_min, max=a_max)

    return corr_mat


def compute_rdm(X, method="correlation"):
    assert method in ["correlation", "euclidean"]
    if method == "euclidean":
        rdm = squareform(pdist(X, metric="euclidean"))
    else:
        rsm = correlation_matrix(X)
        rdm = 1 - rsm

    return rdm


#%%

""" Compute RDMs for two embeddings and plot them. We have not clustered based on the category / class names yet. This would give
an even nicer looking embedding! """

from object_dimensions.utils.utils import load_sparse_codes

dnn_path = "/LOCAL/fmahner/object_dimensions/results/sslab/deep/20mio/sslab/100/16384/2.0/0.25/0/params/pruned_q_mu_epoch_2000.txt"
human_path = "/LOCAL/fmahner/object_dimensions/results/sslab/vgg16_bn/4mio/behavior/sslab/100/256/0.5/0.5/2/params/pruned_q_mu_epoch_1200.txt"

dnn_embedding, dnn_var = load_sparse_codes(dnn_path, with_var=True)
human_embedding, human_var = load_sparse_codes(human_path, with_var=True)

img_root = "/LOCAL/fmahner/object_dimensions/data/image_data/images12_plus"
image_filenames, indices = load_image_data(img_root, filter_behavior=True)

# TODO Maybe make an additional assert statement to check if the number of images in the embedding match the number of loaded images

dnn_embedding = dnn_embedding[indices]
dnn_var = dnn_var[indices]

method = "correlation"

rsm_1 = get_rdm(human_embedding, method)
rsm_2 = get_rdm(dnn_embedding, method)


#%%
from scipy.stats import pearsonr

tril_inds = np.tril_indices(len(rsm_1), k=-1)
tril_1 = rsm_1[tril_inds]
tril_2 = rsm_2[tril_inds]
rho = pearsonr(tril_1, tril_2)[0].round(3)
print(f"\nCorrelation between RSMs: {rho:.3f}\n")

#%%


rsm_1_km = kmeans(rsm_1, 2)
rsm_2_km = kmeans(rsm_2, 2)

rsm_1_km = normalise_rdm(rsm_1_km)
rsm_2_km = normalise_rdm(rsm_1_km)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))


ax1.imshow(rsm_1_km, cmap="viridis_r")
ax1.set_title("Behavior")
ax2.imshow(rsm_2_km, cmap="viridis_r")
ax2.set_title("Deep CNN")

plt.tight_layout()
# fig.savefig("../../../plots/rdm_comparison_vice_behavior_vice_triplets_20mio.png", dpi=150)
# %%
