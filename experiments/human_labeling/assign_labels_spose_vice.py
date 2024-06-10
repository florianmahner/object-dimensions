import numpy as np
from object_dimensions.utils import load_sparse_codes
from scipy.stats import pearsonr
import pandas as pd

path_vice = (
    "./results/behavior/variational/4.12mio/sslab/150/256/1.0/21/params/parameters.npz"
)
path_spose = "./data/misc/spose_embedding_66d_sorted.txt"


weights_vice = load_sparse_codes(path_vice)
weights_spose = load_sparse_codes(path_spose)


labels = np.loadtxt("./data/misc/labels_short.txt", dtype=str, delimiter=",")


def compare_modalities(weights_vice, weights_spose, duplicates=True):
    """Compare the Human behavior embedding to the VGG embedding by correlating them!"""
    assert (
        weights_spose.shape[0] == weights_vice.shape[0]
    ), "\nNumber of items in weight matrices must align.\n"

    dim_a = weights_spose.shape[1]
    dim_b = weights_vice.shape[1]

    corrs_between_modalities = np.zeros(dim_b)
    mod2_dims = []

    for dim_idx_1, weight_1 in enumerate(weights_vice.T):
        # Correlate modality 1 with all dimensions of modality 2
        corrs = np.zeros(dim_a)
        for dim_idx_2, weight_2 in enumerate(weights_spose.T):
            corrs[dim_idx_2] = pearsonr(weight_1, weight_2)[0]

        if duplicates:
            mod2_dims.append(np.argmax(corrs))

        else:
            for ind_mod_2 in np.argsort(-corrs):
                if ind_mod_2 not in mod2_dims:
                    mod2_dims.append(ind_mod_2)
                    break

        corrs_between_modalities[dim_idx_1] = corrs[mod2_dims[-1]]

    return corrs_between_modalities, mod2_dims


corrs, mod2_dims = compare_modalities(weights_vice, weights_spose, duplicates=True)
matches = {i: labels[j] for i, j in enumerate(mod2_dims) if j < 67}
matches = {k: v for k, v in sorted(matches.items(), key=lambda item: item[0])}

df = pd.DataFrame.from_dict(matches, orient="index")
df.to_csv("./data/misc/vice_spose_dim_labels.csv", header=False, index=True)
