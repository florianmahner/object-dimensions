x#%%

import numpy as np
import matplotlib.pyplot as plt
import cv2
from deep_embeddings.utils.utils import compute_rdm, fill_diag
from scipy.stats import rankdata


def pearsonr(u:np.ndarray, v:np.ndarray, a_min:float=-1., a_max:float=1.) -> np.ndarray:
    u_c = u - np.mean(u)
    v_c = v - np.mean(v)
    num = u_c @ v_c
    denom = np.linalg.norm(u_c) * np.linalg.norm(v_c)
    rho = (num / denom).clip(min=a_min, max=a_max)
    return rho

def kmeans(mat, K=50):
    ''' write a function that does k means clustering on an image '''
    # convert to np.float32
    Z = mat.reshape(-1, 1)
    Z = np.float32(Z)
    
    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center= cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((mat.shape))
    return res2


def kmeans_sklearn(img, k):
    """ Write a function that segments a 2d grayscale image using kmeans clustering and sklearn """
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

 


#%%

""" Compute RDMs for two embeddings and plot them. We have not clustered based on the category / class names yet. This would give
an even nicer looking embedding! """

embedding_1 = "../../../results/spose_embedding_66d_sorted.txt"
embedding_2 = "../../../results/weights_things_behavior_8196bs_adaptive_half/params/pruned_q_mu_epoch_5000.txt"
# embedding_2 = "../../../results/weights_vgg_12_512bs/params/pruned_q_mu_epoch_1000.txt"

embedding_1 = np.loadtxt(embedding_1)
embedding_2 = np.loadtxt(embedding_2)

method = 'correlation'

rsm_1 = get_rdm(embedding_1, method)
rsm_2 = get_rdm(embedding_2, method)



#%%
tril_inds = np.tril_indices(len(rsm_1), k=-1)
tril_1 = rsm_1[tril_inds]
tril_2 = rsm_2[tril_inds]
rho = pearsonr(tril_1, tril_2).round(3)
print(f'\nCorrelation between RSMs: {rho:.3f}\n')

#%%


rsm_1 = get_rdm(embedding_1, method)
rsm_2 = get_rdm(embedding_2, method)


rsm_1_km = kmeans(rsm_1, 20)
rsm_2_km = kmeans(rsm_2, 20)

rsm_1_km = normalise_rdm(rsm_1_km)
rsm_2_km = normalise_rdm(rsm_1_km)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))


ax1.imshow(rsm_1_km, cmap='viridis_r')
ax1.set_title("Behavior")
ax2.imshow(rsm_2_km, cmap='viridis_r')
ax2.set_title("Deep CNN")

plt.tight_layout()
# fig.savefig("../../../plots/rdm_comparison_vice_behavior_vice_triplets_20mio.png", dpi=150)
# %%

