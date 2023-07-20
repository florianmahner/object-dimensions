import argparse
import torch
import toml
import numpy as np
import matplotlib.pyplot as plt
from main import build_model
from object_dimensions.utils.utils import load_image_data
from object_dimensions.core.pruning import NormalDimensionPruning


checkpoint = "./embedding/1/checkpoints/epoch_22800.tar"
checkpoint = torch.load(checkpoint)

config = "./embedding/1/config.toml"
config = toml.load(config)
args = argparse.Namespace(**config)


model, _ = build_model(args)
model.load_state_dict(checkpoint["model_state_dict"])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

nonzeros = np.arange(0, 10_000, 100)
dims = []

# for nonzero in nonzeros:
for nonzero in [45]:
    model.non_zero_weights = nonzero

    params = model.sorted_pruned_params()
    q_mu = params["pruned_q_mu"]
    dim = q_mu.shape[1]
    dims.append(dim)

    print("Num dimensions {} nonzero {}".format(dim, nonzero))

breakpoint()

# fig, ax = plt.subplots()
# plt.plot(nonzeros, dims)
# plt.xlabel("Pruning Threshold (# nonzero objects)")
# plt.ylabel("Dimensions")
# plt.savefig("pruned_dimensions.png", dpi=300, bbox_inches="tight", pad_inches=0.1)


# Last dim sum 44.92301858623978
# Check the number of dimensions when reducing the embedding to the 1854 objects!
q_mu = model.q_mu.q_mu.data
q_logvar = model.q_logvar.q_logvar.data


thresh = 44.92301858623978
all_dims = q_mu.shape[1]
for dim in q_mu.T:
    dim = np.maximum(dim, 0.0)
    dim_sum = dim.sum()
    if dim_sum < thresh:
        all_dims -= 1


print("All dims {}".format(all_dims))

breakpoint()


images, indices = load_image_data(
    "./data/image_data/images12_plus", filter_behavior=True
)
n_objects = len(images)

model.q_mu.q_mu.data = q_mu[indices]
model.q_logvar.q_logvar.data = q_logvar[indices]

model.pruner = NormalDimensionPruning(n_objects, 0.0)

model.non_zero_weights = 5

params = model.sorted_pruned_params()
q_mu = params["pruned_q_mu"]
dim = q_mu.shape[1]
print("Num dimensions {} nonzero {}".format(dim, 5))
