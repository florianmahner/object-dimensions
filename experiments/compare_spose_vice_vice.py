from object_dimensions.utils import load_sparse_codes
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


def compare_modalities(weights_vice, weights_spose, duplicates=False, not_sorted=False):
    """Compare the Human behavior embedding to the VGG embedding by correlating them!"""
    assert (
        weights_spose.shape[0] == weights_vice.shape[0]
    ), "\nNumber of items in weight matrices must align.\n"

    dim_human = weights_vice.shape[1]
    dim_dnn = weights_spose.shape[1]

    if dim_human < dim_dnn:
        dim_smaller = dim_human
        dim_larger = dim_dnn
        mod_1 = weights_vice
        mod_2 = weights_spose

    else:
        dim_smaller = dim_dnn
        dim_larger = dim_human
        mod_1 = weights_spose
        mod_2 = weights_vice

    corrs_between_modalities = np.zeros(dim_smaller)
    mod2_dims = []

    for dim_idx_1, weight_1 in enumerate(mod_1.T):
        # Correlate modality 1 with all dimensions of modality 2
        corrs = np.zeros(dim_larger)
        for dim_idx_2, weight_2 in enumerate(mod_2.T):
            corrs[dim_idx_2] = pearsonr(weight_1, weight_2)[0]

        if duplicates:
            mod2_dims.append(np.argmax(corrs))

        else:
            for ind_mod_2 in np.argsort(-corrs):
                if ind_mod_2 not in mod2_dims:
                    mod2_dims.append(ind_mod_2)
                    break

        corrs_between_modalities[dim_idx_1] = corrs[mod2_dims[-1]]

    # Sort the dimensions based on highest correlations!
    mod1_dims_sorted = np.argsort(-corrs_between_modalities)
    mod2_dims_sorted = np.asarray(mod2_dims)[mod1_dims_sorted]
    corrs = corrs_between_modalities[mod1_dims_sorted]

    if not_sorted:
        return mod1_dims_sorted, mod2_dims_sorted, corrs_between_modalities

    else:
        return mod1_dims_sorted, mod2_dims_sorted, corrs


path_vice = (
    "./results/behavior/variational/4.12mio/sslab/150/256/1.0/21/params/parameters.npz"
)
path_spose_martin = "./data/misc/spose_embedding_66d_sorted.txt"
path_spose = "./results/behavior/deterministic/4.12mio/sslab/300/256/0.0037/0.0037/1/params/parameters.npz"

weights_spose_martin = load_sparse_codes(path_spose_martin)
weights_vice = load_sparse_codes(path_vice)
weights_spose = load_sparse_codes(path_spose)


# Compare the modalities
# mod1_dims_sorted, mod2_dims_sorted, corrs = compare_modalities(
#     weights_spose_martin, weights_spose, duplicates=False, not_sorted=False
#     )

# # Compare the modalities
mod1_dims_sorted, mod2_dims_sorted, corrs = compare_modalities(
    weights_spose_martin, weights_vice, duplicates=False, not_sorted=False
)

# Plot the correlations using seaborn
fig, ax = plt.subplots()
sns.lineplot(x=range(len(mod1_dims_sorted)), y=corrs, ax=ax)
ax.set_xlabel("Dimension")
ax.set_ylabel("Pearson r")
ax.set_title("Correlation between VICE and VICE (Lukas)")
# ax.set_title("Correlation between SPoSE and SPOSE (Martin's embedding)")

# fig.savefig("./results/plots/spose_spose_correlations.png", dpi=300, bbox_inches="tight")
fig.savefig("./results/plots/vice_spose_correlations.png", dpi=300, bbox_inches="tight")


labels = np.loadtxt("./data/misc/labels_short.txt", dtype=str, delimiter=",")

breakpoint()
