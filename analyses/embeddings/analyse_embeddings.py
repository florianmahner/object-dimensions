""" 
1. Extract the image representations from DNN representations of the reference images i.e. the ones with behavior!
2. The Behavior embeddding is already only based on these images

-> Compare both of these RDMs (i.e. we build a subset from the VGG embedding, so that it then also only is of shape 1854 x P
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
from deep_embeddings.utils import compute_rdm, fill_diag
from scipy.stats import rankdata

def get_rdm(embedding):
    embedding = np.maximum(0, embedding)
    method = "correlation"
    rsm = compute_rdm(embedding, method)
    rsm = fill_diag(rsm)
    rsm = rankdata(rsm).reshape(rsm.shape)

    return rsm


# behavior_embedding = np.loadtxt("../../learned_embeddings/spose_embedding_66d_sorted.txt")
behavior_embedding = np.loadtxt("../../learned_embeddings/weights_things_behavior_256bs/params/pruned_q_mu_epoch_3000.txt")
vgg_embedding = np.loadtxt("../../learned_embeddings/weights_vgg_12_512bs/params/pruned_q_mu_epoch_150.txt")


# NOTE This loads the filenames in the same order as they have been extracted from VGG features
filenames = glob.glob("../../../THINGS/image_data/images12/*/*")
ref_indices = [i for i, s in enumerate(filenames) if 'b.jpg' in s]
vgg_embedding = vgg_embedding[ref_indices, :]


rsm_behavior = get_rdm(behavior_embedding)
rsm_vgg = get_rdm(vgg_embedding)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(rsm_behavior, cmap="viridis")
ax1.set_title("Behavior VICE")
ax2.imshow(rsm_vgg, cmap="viridis")
ax2.set_title("VGG VICE")

plt.tight_layout()
fig.savefig("../plots/rdm_comparison_behavior_vgg_12images", dpi=150)


