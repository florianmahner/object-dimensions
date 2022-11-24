#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import matplotlib.gridspec as gridspec

from deep_embeddings.utils.utils import load_sparse_codes, filter_embedding_by_behavior, load_image_data
from deep_embeddings import build_triplet_dataset
from deep_embeddings import Embedding as model

parser = argparse.ArgumentParser(description='Jackknife analysis')
parser.add_argument("--dnn_path", type=str, help="Path to the human embedding")
parser.add_argument("--human_path", type=str, help="Path to the human embedding")


def compute_softmax_decisions(q_mu, q_var, indices, device):
    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices)
    indices = indices.type("torch.LongTensor")
    indices = indices.to(device)
    ind_i, ind_j, ind_k = indices.unbind(1)
    
    # Reparametrize an embedding
    torch.manual_seed(0) # fix the seed so that this is not stochastic!

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


def find_diverging_triplets(q_mu, q_var, val_loader, device):
    n_val = len(val_loader)
    softmax_decisions = []
    ooo_indices = []

    q_mu = torch.tensor(q_mu).to(device)
    q_var = torch.tensor(q_var).to(device)

    for k, indices in enumerate(val_loader):
        print("Batch {}/{}".format(k, n_val), end="\r")
        softmax = compute_softmax_decisions(q_mu, q_var, indices, device)
        softmax = softmax[-1] # This is the odd one out probability
    
        # Store the softmax decisions for each index.
        softmax_decisions.append(softmax.detach().cpu().numpy())
        ooo_indices.append(indices.detach().cpu().numpy())

    softmax_decisions = np.concatenate(softmax_decisions)
    ooo_indices = np.concatenate(ooo_indices).astype(int)

    return softmax_decisions, ooo_indices


def jackknife(q_mu, q_var, ooo_indices, device):    
    """ This function computes the jackknife analysis for a given embedding"""
    q_mu = torch.tensor(q_mu).to(device)
    q_var = torch.tensor(q_var).to(device)

    n_dims = q_mu.shape[1]
    soft_max_diff = np.ones(len(ooo_indices)) * float("inf")
    most_important_dim = np.zeros(len(ooo_indices), dtype=int)

    for i in range(n_dims):
        # Take all elements except the i-th embedding dimension

        q_mu_i = torch.cat([q_mu[:,0:i], q_mu[:, i+1:]], dim=1)
        q_var_i = torch.cat([q_var[:, 0:i], q_var[:, i+1:]], dim=1)

        # Compute the softmax decisions
        softmax_per_batch = compute_softmax_decisions(q_mu_i, q_var_i, ooo_indices, device)
        softmax_per_batch = softmax_per_batch[-1] # This is the odd one out probability

        for s, softmax in enumerate(softmax_per_batch):
            if softmax < soft_max_diff[s]:
                soft_max_diff[s] = softmax
                most_important_dim[s] = i
        
    return soft_max_diff, most_important_dim


if __name__ == "__main__":
    args = parser.parse_args()

    args.human_weights = "./results/sslab/vgg16_bn/4mio/behavior/sslab/100/256/0.5/0.5/2/params/pruned_q_mu_epoch_400.txt"
    args.dnn_weights = "./results/50mio/16396/0.4/0.25/1.0/0.5/42/params/pruned_q_mu_epoch_5200.txt"

    args.human_var = "./results/sslab/vgg16_bn/4mio/behavior/sslab/100/256/0.5/0.5/2/params/pruned_q_var_epoch_400.txt"
    args.dnn_var = "./results/50mio/16396/0.4/0.25/1.0/0.5/42/params/pruned_q_var_epoch_5200.txt"

    args.img_root = "./data/image_data/images12"

    _, _, image_filenames = load_image_data(args.img_root)

    human_weights = load_sparse_codes(args.human_weights)

    print(human_weights.shape)
    dnn_weights = load_sparse_codes(args.dnn_weights)

    human_var = load_sparse_codes(args.human_var)
    dnn_var = load_sparse_codes(args.dnn_var)


    # Filter out images without behavioral data
    dnn_weights, ref_images = filter_embedding_by_behavior(dnn_weights, image_filenames)
    dnn_var, ref_images = filter_embedding_by_behavior(dnn_var, image_filenames)


    assert len(human_weights) == len(dnn_weights),  "Embeddings have different shapes! (i.e. different number of images)"

    triplet_path = "./data/triplets_behavior"
    # If all data on GPU, num workers need to be 0 and pin memory false
    device  = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
    train_dataset, val_dataset = build_triplet_dataset(triplet_path, device)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Jackknife analysis
    softmax_human, indices_human = find_diverging_triplets(human_weights, human_var, val_loader, device)
    softmax_dnn, indices_dnn = find_diverging_triplets(dnn_weights, dnn_var, val_loader, device)


    # Find the indices of the most diverging softmax choices!
    softmax_diff = np.abs(softmax_human - softmax_dnn)


    # Sort the indices by the softmax difference
    topk = 4
    sort_indices = np.argsort(-softmax_diff)[:topk]

    # Find the indices of the most diverging softmax choices!
    ooo_human = indices_human[sort_indices]
    ooo_dnn = indices_dnn[sort_indices]

    # Find the most diverging dimensions
    soft_max_diff_human, most_important_dim_human = jackknife(human_weights, human_var, ooo_human, device)
    soft_max_diff_dnn, most_important_dim_dnn = jackknife(dnn_weights, dnn_var, ooo_dnn, device)

    plot_dir = "./jackknife"
    os.makedirs(plot_dir, exist_ok=True)


    for i, (triplet, important_human, important_dnn) in enumerate(zip(ooo_human,most_important_dim_human, most_important_dim_dnn)):

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ctr, img_index in enumerate(triplet):
            img = Image.open(ref_images[img_index])
            img = img.resize((224, 224))

            axes[ctr].imshow(img)
            axes[ctr].axis("off")

        plt.savefig(os.path.join(plot_dir, "triplet_{}.png".format(i)))
        plt.tight_layout()
        fig, axes = plt.subplots(2, 3, figsize=(15, 5))

        human = human_weights[:, important_human]
        human = np.argsort(-human)[:6]

        fig = plt.figure(figsize=(6,4))
        gs1 = gridspec.GridSpec(2, 3)
        gs1.update(wspace=0.025, hspace=0) # set the spacing between axes. 

        for ctr, human_idx in enumerate(human):
            ax = plt.subplot(gs1[ctr])
            img = Image.open(ref_images[human_idx])
            img = img.resize((224, 224))
            
            ax.imshow(img)
            ax.axis("off")

        plt.savefig(os.path.join(plot_dir, "human_{}.png".format(i)))


        dnn = dnn_weights[:, important_dnn]
        dnn = np.argsort(-dnn)[:6]

        fig = plt.figure(figsize=(6,4))
        gs1 = gridspec.GridSpec(2, 3)
        gs1.update(wspace=0.025, hspace=0) # set the spacing between axes. 
        
        for ctr, dnn_idx in enumerate(dnn):
            ax = plt.subplot(gs1[ctr])
            img = Image.open(ref_images[dnn_idx])
            img = img.resize((224, 224))
            
            ax.imshow(img)
            ax.axis("off")

        plt.savefig(os.path.join(plot_dir, "dnn_{}.png".format(i)))
