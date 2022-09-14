# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import thingsvision
import thingsvision.vision as vision

# %%

embedding = np.load('../embeddings/vice_embedding_16mio.npy')
embedding = np.maximum(embedding, 0) # NOTE Positivity constraint

rdm_embedding = vision.compute_rdm(embedding, method='correlation')
fig, ax = plt.subplots()
ax.imshow(rankdata(rdm_embedding).reshape(rdm_embedding.shape), cmap='cividis')
fig.savefig('../plots/embedding_rdm.png', dpi=300)


gt_similarity = np.load('/LOCAL/fmahner/THINGS/vgg_features6/features.npy')
rdm_dnn = vision.compute_rdm(gt_similarity, method='correlation')
fig, ax = plt.subplots()
ax.imshow(rankdata(rdm_dnn).reshape(rdm_dnn.shape), cmap='cividis')
fig.savefig('../plots/vgg_rdm.png', dpi=300)


corr_rdms = vision.correlate_rdms(rdm_embedding, rdm_dnn)

print('Correlation between rdms', corr_rdms)


# %%

feat = gt_similarity
feat = (feat - np.mean(feat, axis=0)) / np.std(feat, axis=0)
S = feat @ feat.T / 4096

# %%
import seaborn as sns
sns.histplot(S, bins='auto')


# %%