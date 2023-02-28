import numpy as np
from scipy.stats import pearsonr
import glob
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd


base_dir = "./results/reproducibility/"
files = glob.glob(base_dir + "**/parameters.npz", recursive=True)
n_dimensions = 500
n_embeddings = len(files)


# Take one base model (original). Check reproducibility to all other models
# Concatenate all embedfinhgs on one dimension?

embeddings = []
min_val = float("inf")
idx = 0
n_pruned = 0
for i, f in enumerate(files):
    params = np.load(f)
    q_mu = params['q_mu']
    q_mu = np.maximum(0, q_mu)
    # Sort dimensions by sum in order!
    ind = np.argsort(-q_mu.sum(0))
    q_mu = q_mu[:,ind]
    embeddings.append(q_mu)

    val_loss = params['val_loss'][-1]
    if val_loss < min_val:
        min_val = val_loss
        idx = i
        n_pruned = params['pruned_q_mu'].shape[-1] 

# Move the one with the lowest validation loss to the front
embeddings.insert(0, embeddings.pop(idx))
n_objects = q_mu.shape[0]

odd_mask = np.arange(n_objects) % 2 == 1
even_mask = np.arange(n_objects) % 2 == 0


print("Min Run", files[idx])

breakpoint()


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


data = pd.DataFrame({"Dimension": np.arange(n_dimensions), "Reproducibility": reliability_per_dim})
ax = sns.lineplot(x="Dimension", y="Reproducibility", data=data, color="black")

# ax = sns.plot(reliability_per_dim, color="black")
ax.set_xlabel("Dimension number")
ax.set_ylabel("Reproducibility of dimensions (Pearson's r)".format(n_embeddings))
ax.set_title("N = {} runs, Batch Size = 16384".format(n_embeddings))
# Draw a vertical line at index n_pruned
ax.axvline(x=n_pruned, color="red", linestyle="--")

plt.tight_layout()
plt.savefig(os.path.join("./results", "plots", "reproducibility_across_dimensions.png"), dpi=300, bbox_inches="tight", pad_inches=0.1)