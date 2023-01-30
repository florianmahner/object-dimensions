import numpy as np
from scipy.stats import pearsonr
import glob
import matplotlib.pyplot as plt


base_dir = "./results/reproducibility/"
files = glob.glob(base_dir + "**/parameters.npz", recursive=True)
n_dimensions = 500
n_embeddings = len(files)


# Take one base model (original). Check reproducibility to all other models
# Concatenate all embedfinhgs on one dimension?

embeddings = []
for f in files:
    params = np.load(f)
    q_mu = params['q_mu']
    q_mu = np.maximum(0, q_mu)
    # Sort dimensions by sum in order!
    ind = np.argsort(-q_mu.sum(0))
    q_mu = q_mu[:,ind]
    embeddings.append(q_mu)

n_objects = q_mu.shape[0]

odd_mask = np.arange(n_objects) % 2 == 1
even_mask = np.arange(n_objects) % 2 == 0


reliability_per_dim = []

print("Number of files: {}".format(n_embeddings))

for i in range(n_dimensions):
    embed_i = embeddings[0][:, i] # this is the base embedding

    
    k_per_dim = []
    for k in range(1, n_embeddings):

        odd_corrs = []

        for j in range(n_dimensions):        
            embed_jk = embeddings[k][:, j]
            odd_correlation = pearsonr(embed_i[odd_mask], embed_jk[odd_mask])[0]
            odd_corrs.append(odd_correlation)


        # Take the clostest match and correlate with the even embedding
        closest_match = np.argmax(odd_corrs)
        embed_compare = embeddings[k][:, closest_match]
        even_corr = pearsonr(embed_i[even_mask], embed_compare[even_mask])[0]

        k_per_dim.append(even_corr)

    
    z_transformed = np.arctanh(np.array(k_per_dim))
    average = np.mean(z_transformed)
    back_transformed = np.tanh(average)
    reliability_per_dim.append(back_transformed)


    print("Reliability for dimension {} is {}".format(i, back_transformed))


fig, ax = plt.subplots()
ax.plot(reliability_per_dim)
ax.set_xlabel("Dimension number")
ax.set_ylabel("Reproducibility of dimensions (n = {} runs)".format(n_embeddings))
plt.tight_layout()
plt.savefig("reproducibility_across_dimensions.png", dpi=300, bbox_inches="tight", pad_inches=0.1)