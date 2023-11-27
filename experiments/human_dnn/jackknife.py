#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" This script computes the jackknife analysis for the deep embeddings. Jackknife 
means that we remove one dimension at a time and compute the softmax decision for the odd one out triplet.
We then look at the dimension that has the largest impact on the softmax decision by removing it (jackknifing it).
and evaluate the impact of the dimension on the softmax decision. """


import torch
import os
import torch.nn.functional as F
import numpy as np
import seaborn as sns
from scipy.stats import rankdata
import itertools
import pickle
import matplotlib.pyplot as plt
from tomlparse import argparse
from object_dimensions import build_triplet_dataset
from object_dimensions import VariationalEmbedding as model
from object_dimensions.utils import (
    load_image_data,
    create_path_from_params,
    load_sparse_codes,
)
from experiments.human_dnn.plot_jackknife import plot_grid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Jackknife analysis for the deep embeddings",
    )
    parser.add_argument("--human_path", type=str, default="data/human_data.pkl")
    parser.add_argument("--dnn_path", type=str, default="data/dnn_data.pkl")
    parser.add_argument("--triplet_path", type=str, default="data/triplets.pkl")
    parser.add_argument("--img_root", type=str, default="data/images")
    parser.add_argument(
        "--comparison", type=str, default="dnn", choices=["dnn", "human"]
    )
    parser.add_argument("--run_jackknife", action="store_true", default=False)
    parser.add_argument("--plot", action="store_true", default=False)
    return parser.parse_args()


def compute_softmax_per_batch(q_mu, q_var, indices, device):
    """This function extracts the embedding vectors at the indices of the most diverging triplets and computes the
    softmax decision for each of them"""

    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices)

    torch.manual_seed(0)  # fix the seed so that this is not stochastic!
    embedding = model.reparameterize_sslab(q_mu, q_var)

    # nll, acc = calculate_likelihood(embedding, indices)
    # indices = indices.type("torch.LongTensor").to(device)
    indices = indices.unbind(1)
    ind_i, ind_j, ind_k = indices
    embedding_i = F.relu(embedding[ind_i])
    embedding_j = F.relu(embedding[ind_j])
    embedding_k = F.relu(embedding[ind_k])

    sim_ij = torch.einsum("ij,ij->i", embedding_i, embedding_j)
    sim_ik = torch.einsum("ij,ij->i", embedding_i, embedding_k)
    sim_jk = torch.einsum("ij,ij->i", embedding_j, embedding_k)

    # Compute the log softmax loss, i.e. we just look at the argmax anyways!
    sims = torch.cat([sim_ij, sim_ik, sim_jk]).view(3, -1)  # faster than torch.stack

    log_softmax = F.log_softmax(sims, dim=0)  # i.e. 3 x batch_size

    # Theis the most similar (sim_ij), i.e. the argmax, k is the ooo.
    log_softmax_ij = log_softmax[0]

    # Compute the cross entropy loss -> we dont need to one hot encode the batch
    # We just use the similarities at index 0, all other targets are 0 and redundant!
    nll = torch.mean(-log_softmax_ij)

    # Compute the accuracy
    acc = torch.mean((torch.argmax(log_softmax, dim=0) == 0).float())
    softmax = F.softmax(sims, dim=0).T  # i.e. 3 x batch_size

    return softmax


def compute_softmax_decisions(q_mu, q_var, val_loader, device):
    """We compute the softmax choices for all triplets"""
    q_mu = torch.tensor(q_mu).to(device)
    q_var = torch.tensor(q_var).to(device)

    n_batch = len(val_loader)
    n_val = len(val_loader.dataset)
    bs = val_loader.batch_size

    softmax_decisions = torch.zeros(n_val, 3, device=device)
    ooo_indices = torch.zeros(n_val, 3, device=device)

    for k, indices in enumerate(val_loader):
        print("Batch {}/{}".format(k, n_batch), end="\r")
        softmax = compute_softmax_per_batch(q_mu, q_var, indices, device)
        # Add the softmax at the correct index
        softmax_decisions[k * bs : (k + 1) * bs] = softmax.detach()
        ooo_indices[k * bs : (k + 1) * bs] = indices.detach()

    softmax_decisions = softmax_decisions.detach().cpu().numpy()
    ooo_indices = ooo_indices.detach().cpu().numpy()
    ooo_indices = ooo_indices.astype(int)

    return softmax_decisions, ooo_indices


def find_diverging_triplets(softmax_human, softmax_dnn, topk=12):
    """We ranksort the softmax decisions of the human and the dnn and find the triplets that are most
    diverging, e.g. where the rank is high in one and low in the other."""
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


def jackknife(q_mu, q_var, triplet_indices, device, ooo_index=0):
    """This function computes the jackknife analysis for a given embedding"""
    q_mu = torch.tensor(q_mu).to(device)
    q_var = torch.tensor(q_var).to(device)

    n_dims = q_mu.shape[1]
    softmax_diff = np.ones((len(triplet_indices), n_dims)) * float("inf")

    # Without jackknifing for that triplet
    softmax_default = compute_softmax_per_batch(q_mu, q_var, triplet_indices, device)
    softmax_default = softmax_default[:, ooo_index]
    softmax_default = softmax_default.detach().cpu().numpy()

    for i in range(n_dims):
        # Take all elements except the i-th embedding dimension
        q_mu_i = torch.cat([q_mu[:, 0:i], q_mu[:, i + 1 :]], dim=1)
        q_var_i = torch.cat([q_var[:, 0:i], q_var[:, i + 1 :]], dim=1)

        # Compute the softmax decisions
        softmax_per_batch = compute_softmax_per_batch(
            q_mu_i, q_var_i, triplet_indices, device
        )

        # This is the odd one out probability (at the index of the odd one out)
        softmax_per_batch = softmax_per_batch[:, ooo_index]
        softmax_per_batch = softmax_per_batch.detach().cpu().numpy()
        softmax_diff[:, i] = np.abs(softmax_per_batch - softmax_default)

    most_important_dim = np.argmax(softmax_diff, axis=1)

    return softmax_default, softmax_diff, most_important_dim


def build_dataloader(triplet_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_dataset = build_triplet_dataset(triplet_path, device=device)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False, num_workers=8
    )
    return val_loader


def calculate_likelihood(embedding, indices):
    """Compute the negative log likelihood of the data given the embedding"""
    indices = indices.unbind(1)
    ind_i, ind_j, ind_k = indices

    embedding_i = F.relu(embedding[ind_i])
    embedding_j = F.relu(embedding[ind_j])
    embedding_k = F.relu(embedding[ind_k])

    sim_ij = torch.einsum("ij,ij->i", embedding_i, embedding_j)
    sim_ik = torch.einsum("ij,ij->i", embedding_i, embedding_k)
    sim_jk = torch.einsum("ij,ij->i", embedding_j, embedding_k)

    # Compute the log softmax loss, i.e. we just look at the argmax anyways!
    sims = torch.cat([sim_ij, sim_ik, sim_jk]).view(3, -1)  # faster than torch.stack

    log_softmax = F.log_softmax(sims, dim=0)  # i.e. 3 x batch_size

    # Theis the most similar (sim_ij), i.e. the argmax, k is the ooo.
    log_softmax_ij = log_softmax[0]

    # Compute the cross entropy loss -> we dont need to one hot encode the batch
    # We just use the similarities at index 0, all other targets are 0 and redundant!
    nll = torch.mean(-log_softmax_ij)

    # Compute accuracy for that batch
    triplet_choices = torch.argmax(log_softmax, 0)
    triplet_accuracy = torch.sum(triplet_choices == 0) / len(triplet_choices)

    return nll, triplet_accuracy


def compute_jackknife(
    val_loader,
    human_weights,
    human_var,
    dnn_weights,
    dnn_var,
    plot_dir,
    device,
    topk=12,
):
    save_dict = dict()

    softmax_human, triplets = compute_softmax_decisions(
        human_weights, human_var, val_loader, device
    )
    softmax_dnn, triplets = compute_softmax_decisions(
        dnn_weights, dnn_var, val_loader, device
    )

    # All combinations of softmax probabilities for the odd one out
    numbers = [0, 1, 2]
    combinations = list(itertools.product(numbers, repeat=2))

    # Map each of the combs to a string 0=k, 1=i, 2=j for a list of combinations of pairs of 2
    # When the softmax at position 0=i is the highest, then the model identifies object k as the odd one out
    identifiers = []
    lookup_table = ["k", "j", "i"]
    for comb in combinations:
        str_1, str_2 = [lookup_table[i] for i in comb]
        identifiers.append((str_1, str_2))

    for comb, iden in zip(combinations, identifiers):
        str_1, str_2 = iden
        index_1, index_2 = comb

        softmax_human_choice = softmax_human[:, index_1]
        softmax_dnn_choice = softmax_dnn[:, index_2]

        most_diverging_indices = find_diverging_triplets(
            softmax_human_choice, softmax_dnn_choice, topk
        )

        # For each triplet, find the most important dimension by jackknifing it iteratively
        jackknife_results = dict()
        for name, diverging_triplets in most_diverging_indices.items():
            interesting_triplets = triplets[diverging_triplets]

            softmax_default_human, softmax_diff_human, important_human = jackknife(
                human_weights,
                human_var,
                interesting_triplets,
                device,
                ooo_index=index_1,
            )

            softmax_default_dnn, softmax_diff_dnn, important_dnn = jackknife(
                dnn_weights, dnn_var, interesting_triplets, device, ooo_index=index_2
            )

            jackknife_results[name] = {
                "triplets": interesting_triplets,
                "softmax_diff_human": softmax_diff_human,
                "softmax_default_human": softmax_default_human,
                "softmax_diff_dnn": softmax_diff_dnn,
                "softmax_default_dnn": softmax_default_dnn,
                "dims_human": important_human,
                "dims_dnn": important_dnn,
            }

        iden = "human_{}_dnn_{}".format(str_1, str_2)
        save_dict[iden] = jackknife_results

    save_dict["softmax_human"] = softmax_human
    save_dict["softmax_dnn"] = softmax_dnn
    save_dict["triplets"] = triplets
    save_dict["human_weights"] = human_weights
    save_dict["dnn_weights"] = dnn_weights

    # Save the dict into the plot directory
    with open(os.path.join(plot_dir, "jackknife.pkl"), "wb") as f:
        pickle.dump(save_dict, f)


def main(
    human_path, dnn_path, img_root, triplet_path, topk=12, run_jackknife=True, plot=True
):
    """Compare the human and DNN embeddings"""
    # If all data on GPU, num workers need to be 0 and pin memory false
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    dnn_weights, dnn_var = load_sparse_codes(dnn_path, with_var=True, relu=True)
    human_weights, human_var = load_sparse_codes(human_path, with_var=True, relu=True)

    # triplet_path_dnn = "./data/triplets/vgg16_bn/classifier.3/triplets_20mio"
    triplet_path = "./data/triplets/behavior"

    val_loader = build_dataloader(triplet_path)

    plot_dir = create_path_from_params(dnn_path, "analyses", "jackknife")
    print("Save to '{}'".format(plot_dir))

    # Load image data
    image_filenames, indices = load_image_data(img_root, filter_behavior=True)
    dnn_weights_behavior = dnn_weights[indices]
    dnn_var_behavior = dnn_var[indices]

    if run_jackknife:
        # PLot this for the DNN for all
        softmax_dnn_non_filtered, _ = compute_softmax_decisions(
            dnn_weights_behavior, dnn_var_behavior, val_loader, device
        )
        sns.set(style="whitegrid", context="paper", font_scale=1.2)
        column_labels = {0: "k", 1: "j", 2: "i"}
        colors = {0: "blue", 1: "green", 2: "red"}

        fig, ax = plt.subplots(figsize=(6, 3))
        for k in range(3):
            sns.histplot(
                softmax_dnn_non_filtered[:, k],
                ax=ax,
                bins=100,
                color=colors[k],
                alpha=0.5,
                label=f"{column_labels[k]}",
            )

        ax.set_xlabel("Softmax probability")
        ax.set_ylabel("Count")
        ax.legend(title="Odd one out")

        fig.tight_layout()
        plt.savefig(
            os.path.join(plot_dir, "softmax_histogram_dnn_all.png"),
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.1,
        )
        plt.close(fig)

        dnn_smaller_weights = dnn_weights_behavior[:, :84]
        dnn_smaller_var = dnn_var_behavior[:, :84]

        softmax_dnn_non_filtered, _ = compute_softmax_decisions(
            dnn_smaller_weights, dnn_smaller_var, val_loader, device
        )
        sns.set(style="whitegrid", context="paper", font_scale=1.2)
        column_labels = {0: "k", 1: "j", 2: "i"}
        colors = {0: "blue", 1: "green", 2: "red"}

        fig, ax = plt.subplots(figsize=(6, 3))
        for k in range(3):
            sns.histplot(
                softmax_dnn_non_filtered[:, k],
                ax=ax,
                bins=100,
                color=colors[k],
                alpha=0.5,
                label=f"{column_labels[k]}",
            )

        ax.set_xlabel("Softmax probability")
        ax.set_ylabel("Count")
        ax.legend(title="Odd one out")

        fig.tight_layout()
        plt.savefig(
            os.path.join(plot_dir, "softmax_histogram_dnn.png"),
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.1,
        )
        plt.close(fig)

        val_loader = build_dataloader(triplet_path)
        print("Yay")

    if run_jackknife:
        compute_jackknife(
            val_loader,
            human_weights,
            human_var,
            dnn_weights_behavior,
            dnn_var_behavior,
            plot_dir,
            device,
            topk=topk,
        )

    if plot:
        jackknife_path = os.path.join(plot_dir, "jackknife.pkl")

        # Raise file not found error if the jackknife file does not exist
        if not os.path.exists(jackknife_path):
            raise FileNotFoundError(
                "Jackknife file not found at {}. Consider running the analysis first.".format(
                    jackknife_path
                )
            )

        jackknife_dict = pickle.load(open(jackknife_path, "rb"))
        for key in [
            "human_i_dnn_j",
            "human_j_dnn_i",
            "human_k_dnn_k",
            "human_k_dnn_j",
            "human_k_dnn_i",
        ]:
            # for key in [("human_k_dnn_k")]:
            print("Plotting {}".format(key))

            image_filenames, indices = load_image_data(
                img_root,
                filter_plus=True,
            )
            # NOTE we needed human behavior ones to make the comparison. for plotting we
            # now take the plus images
            dnn_weights_plus = dnn_weights[indices]
            jackknife_dict["dnn_weights"] = dnn_weights_plus

            plot_grid(
                plot_dir,
                image_filenames,
                jackknife_dict,
                key,
                softmax_key="high_both",
                max_topk=5,
            )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.human_path,
        args.dnn_path,
        args.img_root,
        args.triplet_path,
        24,
        args.run_jackknife,
        args.plot,
    )
