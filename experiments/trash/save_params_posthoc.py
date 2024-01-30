import glob
import os
import torch
import numpy as np


def save_params_npz_posthoc(base_dir):
    """Save the parameters.npz file after the experiment has finished"""
    file_list = glob.glob(
        os.path.join(base_dir, "**/checkpoints/*.tar"), recursive=True
    )

    print("Number of confiugrations found: {}".format(len(file_list)))
    if not file_list:
        raise ValueError("No checkpoints files found in {}".format(base_dir))

    for file in file_list:
        # Load the checkpoint and add the parameters to the parameters.npz file
        ckpt = torch.load(file, map_location="cpu")
        params = ckpt["params"]

        # model = torch.load(ckpt["model_state_dict"])
        # pruned_params = model.sorted_pruned_params()
        # # params.update(
        #     pruned_q_mu=pruned_params["pruned_q_mu"], pruned_q_var=pruned_params["pruned_q_var"], embedding=params["embedding"]
        # )

        # params_dir = os.path.join(os.path.dirname(file), "params", "parameters.npz")
        # np.savez(params_dir, **params)


# base_dir = "./results/sslab_final"
base_dir = (
    "./results/sslab_final/deep/vgg16_bn/classifier.3/20.mio/sslab/500/16384/0.55/1.0"
)
save_params_npz_posthoc(base_dir)
