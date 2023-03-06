#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" This script computes the jackknife analysis for the deep embeddings. Jackknife 
means that we remove one dimension at a time and compute the softmax decision for the odd one out triplet.
We then look at the dimension that has the largest impact on the softmax decision by removing it (jackknifing it).
and evaluate the impact of the dimension on the softmax decision. """


import torch
import os
import torch.nn.functional as F
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import rankdata
import seaborn as sns

import matplotlib.gridspec as gridspec

from deep_embeddings import build_triplet_dataset
from deep_embeddings import VariationalEmbedding as model


def compute_softmax_per_batch(q_mu, q_var, indices, device):
    """This function extracts the embedding vectors at the indices of the most diverging triplets and computes the
    softmax decision for each of them"""

    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices)

    indices = indices.type("torch.LongTensor")
    indices = indices.to(device)
    ind_i, ind_j, ind_k = indices.unbind(1)

    # Reparametrize an embedding
    torch.manual_seed(0)  # fix the seed so that this is not stochastic!

    embedding = model.reparameterize_sslab(q_mu, q_var)

    # We add a posivity constraint to this embedding, but this is maybe not necessary?!
    embedding_i = F.relu(embedding[ind_i])
    embedding_j = F.relu(embedding[ind_j])
    embedding_k = F.relu(embedding[ind_k])

    sim_ij = torch.sum(embedding_i * embedding_j, dim=1)
    sim_ik = torch.sum(embedding_i * embedding_k, dim=1)
    sim_jk = torch.sum(embedding_j * embedding_k, dim=1)

    # compute the log softmax loss, i.e. we just look at the argmax anyways!
    sims = torch.stack([sim_ij, sim_ik, sim_jk])
    softmax = F.softmax(sims, dim=0)  # i.e. BS x 3

    return softmax


def compute_softmax_decisions(q_mu, q_var, val_loader, device):
    """ We compute the softmax choices for all triplets """
    n_val = len(val_loader)
    softmax_decisions = []
    ooo_indices = []

    q_mu = torch.tensor(q_mu).to(device)
    q_var = torch.tensor(q_var).to(device)

    for k, indices in enumerate(val_loader):
        print("Batch {}/{}".format(k, n_val), end="\r")
        softmax = compute_softmax_per_batch(q_mu, q_var, indices, device)
        # softmax = softmax[-1] # This is the odd one out probability
        softmax = softmax[
            0
        ]  # this is the similarity of object i,j (k) is the odd one out

        # Store the softmax decisions for each index.
        softmax_decisions.append(softmax.detach().cpu().numpy())
        ooo_indices.append(indices.detach().cpu().numpy())

    softmax_decisions = np.concatenate(softmax_decisions)
    ooo_indices = np.concatenate(ooo_indices).astype(int)

    return softmax_decisions, ooo_indices


def find_diverging_triplets(softmax_human, softmax_dnn, indices, topk=12):
    """We ranksort the softmax decisions of the human and the dnn and find the triplets that are most
    diverging, i.e. where the rank is high in one and low in the other."""
    rank_human = rankdata(softmax_human)
    rank_dnn = rankdata(softmax_dnn)

    high_both = np.mean([rank_human, rank_dnn], axis=0)
    high_both = np.argsort(-high_both)[:topk]

    high_human_low_dnn = np.mean([rank_human, -rank_dnn], axis=0)
    high_human_low_dnn = np.argsort(-high_human_low_dnn)[:topk]

    low_human_high_dnn = np.mean([-rank_human, rank_dnn], axis=0)
    low_human_high_dnn = np.argsort(-low_human_high_dnn)[:topk]

    diverging_indices = dict(
        high_both=high_both,
        high_human_low_dnn=high_human_low_dnn,
        low_human_high_dnn=low_human_high_dnn,
    )

    return diverging_indices


def jackknife(q_mu, q_var, ooo_indices, device):
    """ This function computes the jackknife analysis for a given embedding"""
    q_mu = torch.tensor(q_mu).to(device)
    q_var = torch.tensor(q_var).to(device)

    n_dims = q_mu.shape[1]
    softmax_diff = np.ones((len(ooo_indices), n_dims)) * float("inf")
    most_important_dim = np.zeros(len(ooo_indices), dtype=int)

    softmax_all = np.ones((len(ooo_indices), n_dims))

    # Without jackknifing for that triplet
    softmax_default = compute_softmax_per_batch(q_mu, q_var, ooo_indices, device)[0]
    softmax_default = softmax_default.detach().cpu().numpy()

    for i in range(n_dims):
        # Take all elements except the i-th embedding dimension

        q_mu_i = torch.cat([q_mu[:, 0:i], q_mu[:, i + 1 :]], dim=1)
        q_var_i = torch.cat([q_var[:, 0:i], q_var[:, i + 1 :]], dim=1)

        # Compute the softmax decisions
        softmax_per_batch = compute_softmax_per_batch(
            q_mu_i, q_var_i, ooo_indices, device
        )
        # softmax_per_batch = softmax_per_batch[-1] # This is the odd one out probability (at index k)
        softmax_per_batch = softmax_per_batch[
            0
        ]  # This is the similarity of object i,j (k) is the odd one out

        softmax_per_batch = softmax_per_batch.detach().cpu().numpy()
        softmax_all[:, i] = softmax_per_batch
        softmax_diff[:, i] = np.abs(softmax_per_batch - softmax_default)

    most_important_dim = np.argmin(softmax_all, axis=1)

    return softmax_diff, most_important_dim


def build_dataloader(triplet_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_dataset = build_triplet_dataset(triplet_path, device)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=4
    )

    return val_loader


def plot_bar(decisions, dimensions, out_path="./rose.png"):
    # Get the topk most important dimensions that changed the most for a triplet
    topk = 12
    decisions = decisions[np.argsort(-decisions)][:topk]

    decisions = decisions / decisions.sum()
    decisions = decisions * 1000
    decisions = np.array(decisions, dtype=int)

    dim = [f"Dim {str(i)}" for i in dimensions]
    df = pd.DataFrame({"softmax": decisions, "dim": dim})

    plt.figure(figsize=(6, 8))
    ax = plt.subplot(111, polar=True)
    plt.axis("off")

    lowerLimit = 0
    palette = sns.color_palette("husl", 13)

    # Compute the angle each bar is centered on:
    indexes = list(range(1, len(df.index) + 1))
    width2 = 2 * np.pi / len(df.index)
    angles = [element * width2 for element in indexes]

    # Draw bars
    bars = ax.bar(
        x=angles,
        height=df["softmax"],
        width=0.2,
        bottom=lowerLimit,
        linewidth=2,
        edgecolor="white",
        color=palette,
    )

    # little space between the bar and the label
    labelPadding = 0.1

    # Add labels
    for bar, angle, height, label in zip(bars, angles, df["softmax"], df["dim"]):

        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)

        # Flip some labels upside down
        alignment = ""
        if angle >= np.pi / 2 and angle < 3 * np.pi / 2:
            alignment = "right"
            rotation = rotation + 180
        else:
            alignment = "left"

        # Add a label to the bar right below the bar rotated
        ax.text(
            x=angle,
            y=height + labelPadding,
            s=label,
            rotation=rotation,
            rotation_mode="anchor",
            ha=alignment,
            va="center_baseline",
            fontsize=15,
        )

    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)


def run_jackknife(
    human_weights,
    human_var,
    dnn_weights,
    dnn_var,
    image_filenames,
    triplet_path,
    plot_dir,
    topk=12,
    comparison="dnn",
):
    """ This function runs the jackknife analysis for a given embedding"""

    # TODO Show triplets and dimensions where the DNN is sure about the odd one out but actually wrong!

    if comparison == "human":
        title_1 = "Dimension Human 1"
        title_2 = "Dimension Human 2"
    else:
        title_1 = "Dimension Human"
        title_2 = "Dimension DNN"

    # If all data on GPU, num workers need to be 0 and pin memory false
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    val_loader = build_dataloader(triplet_path)

    softmax_human, indices = compute_softmax_decisions(
        human_weights, human_var, val_loader, device
    )
    softmax_dnn, indices = compute_softmax_decisions(
        dnn_weights, dnn_var, val_loader, device
    )

    # Indices human and indices DNN are the same
    most_diverging_indices = find_diverging_triplets(
        softmax_human, softmax_dnn, indices, topk
    )

    # Replace all keys in most_divering_indices that start with dnn with human
    most_diverging_indices_human = dict()
    for key, value in most_diverging_indices.items():
        most_diverging_indices_human[key.replace("dnn", "human")] = value

    most_diverging_indices = most_diverging_indices_human

    # Where the DNN is sure about the odd one out -> This is currently not meaningul...
    """
    interesting_dnn_indices = rankdata(softmax_dnn, method="max")[0:topk].astype(int)
    interesting_triplets = indices[interesting_dnn_indices]
    softmax_diff = jackknife(human_weights, human_var, interesting_triplets, device)[0]

    # Shape topk x topk
    dims = np.argsort(-softmax_diff, axis=1)[:, :topk]
    
    filtered_diffs = np.zeros((topk, topk))
    for i, diff in enumerate(softmax_diff):

        filtered_diffs[i] = diff[dims[i]]

    # plot_bar(filtered_diffs[0], dims[0], out_path=os.path.join(plot_dir, "rose_0.png"))
    """

    most_important_dim_human = dict()
    most_important_dim_dnn = dict()

    # FOr two embeddings, we have found the most diverging triplets. For these triplets we now iteratively
    # observe the change in softmax probability when removing one dimension at a time. This is the jackknife analysis
    # and gives us a probability measure for each dimension and its importance towards making a decision for a
    # given triplet
    for key, value in most_diverging_indices.items():
        print(f"{key}: {value}")

        interesting_triplets = indices[value]

        softmax_diff, important_human = jackknife(
            human_weights, human_var, interesting_triplets, device
        )
        most_important_dim_human[key] = important_human

        softmax_diff, important_dnn = jackknife(
            dnn_weights, dnn_var, interesting_triplets, device
        )
        most_important_dim_dnn[key] = important_dnn

    for key, value in most_diverging_indices.items():
        print(f"Starting {key}")

        dim_human = most_important_dim_human[key]
        dim_dnn = most_important_dim_dnn[key]
        interesting_triplets = indices[value]

        for i, (triplet, dim_h, dim_d) in enumerate(
            zip(interesting_triplets, dim_human, dim_dnn)
        ):

            fig = plt.figure(figsize=(15, 10))
            gs0 = gridspec.GridSpec(1, 3, figure=fig)
            gs0.update(
                top=0.95, bottom=0.7, left=0.25, right=0.75, wspace=0.0, hspace=0.0
            )

            for ctr, img_index in enumerate(triplet):
                ax = plt.subplot(gs0[ctr])

                img = Image.open(image_filenames[img_index])
                img = img.resize((224, 224))

                ax.imshow(img)
                ax.axis("off")

            plt.subplot(gs0[1]).set_title("Triplet", fontsize=20)

            human = human_weights[:, dim_h]
            human = np.argsort(-human)[:6]

            dnn = dnn_weights[:, dim_d]
            dnn = np.argsort(-dnn)[:6]

            # fig = plt.figure(figsize=(15, 5))
            gs1 = gridspec.GridSpec(2, 3, figure=fig)
            gs1.update(
                top=0.65, bottom=0.1, left=0.05, right=0.45, wspace=0.0, hspace=0.0
            )

            for ctr, human_idx in enumerate(human):
                ax = plt.subplot(gs1[ctr])
                img = Image.open(image_filenames[human_idx])
                img = img.resize((224, 224), Image.Resampling.BILINEAR)
                ax.imshow(img)
                ax.axis("off")

            plt.subplot(gs1[1]).set_title(title_1, fontsize=20)

            gs2 = gridspec.GridSpec(2, 3, figure=fig)
            gs2.update(
                top=0.65, bottom=0.1, left=0.55, right=0.95, wspace=0.0, hspace=0.0
            )

            for ctr, dnn_idx in enumerate(dnn):
                ax = plt.subplot(gs2[ctr])
                img = Image.open(image_filenames[dnn_idx])
                img = img.resize((224, 224))

                ax.imshow(img)
                ax.axis("off")

            plt.subplot(gs2[1]).set_title(title_2, fontsize=20)
            plt.savefig(os.path.join(plot_dir, "{}_{}.png".format(key, i)), dpi=300)
            plt.close()
