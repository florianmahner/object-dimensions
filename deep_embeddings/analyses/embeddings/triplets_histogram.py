#%%
import numpy as np
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt

#%%

# Plotting the triplet distribiution -> Do not delete

triplets = np.load('/LOCAL/fmahner/THINGS/triplets_12_20mio_adaptive/train_90.npy')

plot_distrib = True
if plot_distrib:
    count = Counter(triplets.flatten())
    fig, ax = plt.subplots(1)
    ax.bar(count.keys(), count.values())
    ax.set_xlabel('Image Identifier')
    ax.set_ylabel('Count')

    fig.savefig('histogram_triplets_12_20mio_adaptive.png', dpi=300)

# Plot 2D matrix
n_samples = len(triplets)
n_images =  sum(1 for line in open('/LOCAL/fmahner/THINGS/vgg_bn_features_12/file_names.txt')) 
matrix = np.zeros((n_images, n_images))

for i, tri in enumerate(triplets):

    print(f'{i}/{n_samples}', end='\r')

    idx1, idx2 = tri[:2]
    matrix[idx1, idx2] +=1
    matrix[idx2, idx1] +=1

    idx1, idx2 = tri[1:3]
    matrix[idx1, idx2] +=1
    matrix[idx2, idx1] +=1

    idx1, idx2 = tri[0], tri[2]
    matrix[idx1, idx2] +=1
    matrix[idx2, idx1] +=1


np.save('similarity_mat_12_20mio_adaptive.npy', matrix)


#%%

plot_tri = True

if plot_tri:

    matrix = np.load('similarity_mat_12_20mio_adaptive.npy')

    low_tri = np.tril(matrix) # everything above diag is then zero

    count = Counter(low_tri.flatten())

    fig, ax = plt.subplots(1)
    ax.bar(count.keys(), count.values())
    ax.set_xlabel('Entries in Similarity Matrix')
    ax.set_ylabel('Counts')
    fig.savefig('counts_similarity_mat.png', dpi=300)

    fig, ax = plt.subplots(1)
    ax.imshow(matrix, cmap='gray_r')

    # sns.histplot(low_tri, legend=False, cumulative=False)








# %%
