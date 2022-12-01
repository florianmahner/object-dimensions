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

import matplotlib.gridspec as gridspec

from deep_embeddings import build_triplet_dataset
from deep_embeddings import Embedding as model


def compute_softmax_per_batch(q_mu, q_var, indices, device):
    """ This function extracts the embedding vectors at the indices of the most diverging triplets and computes the 
    softmax decision for each of them"""

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
        softmax = softmax[0] # this is the similarity of object i,j (k) is the odd one out
    
        # Store the softmax decisions for each index.
        softmax_decisions.append(softmax.detach().cpu().numpy())
        ooo_indices.append(indices.detach().cpu().numpy())

    softmax_decisions = np.concatenate(softmax_decisions)
    ooo_indices = np.concatenate(ooo_indices).astype(int)

    return softmax_decisions, ooo_indices


def find_diverging_triplets(softmax_human, softmax_dnn, indices, topk=12):

    # # Find the indices of the most diverging softmax choices!
    # softmax_diff = np.abs(softmax_human - softmax_dnn)

    # # Sort the indices by the softmax difference
    # sort_indices = np.argsort(-softmax_diff)[:topk]


    rank_human = rankdata(softmax_human)
    rank_dnn = rankdata(softmax_dnn)

    high_both = np.mean([rank_human, rank_dnn], axis=0)
    high_both = np.argsort(-high_both)[:topk]

    high_human_low_dnn = np.mean([rank_human, -rank_dnn], axis=0)
    high_human_low_dnn = np.argsort(-high_human_low_dnn)[:topk]

    low_human_high_dnn = np.mean([-rank_human, rank_dnn], axis=0)
    low_human_high_dnn = np.argsort(-low_human_high_dnn)[:topk]

    
    diverging_indices = dict(high_both=high_both, 
                             high_human_low_dnn=high_human_low_dnn, 
                             low_human_high_dnn=low_human_high_dnn)

    return diverging_indices


def jackknife(q_mu, q_var, ooo_indices, device):    
    """ This function computes the jackknife analysis for a given embedding"""
    q_mu = torch.tensor(q_mu).to(device)
    q_var = torch.tensor(q_var).to(device)

    n_dims = q_mu.shape[1]
    softmax_diff = np.ones(len(ooo_indices)) * float("inf")
    most_important_dim = np.zeros(len(ooo_indices), dtype=int)

    softmax_all = np.ones((len(ooo_indices), n_dims)) 


    for i in range(n_dims):
        # Take all elements except the i-th embedding dimension

        q_mu_i = torch.cat([q_mu[:,0:i], q_mu[:, i+1:]], dim=1)
        q_var_i = torch.cat([q_var[:, 0:i], q_var[:, i+1:]], dim=1)

        # Compute the softmax decisions
        softmax_per_batch = compute_softmax_per_batch(q_mu_i, q_var_i, ooo_indices, device)
        # softmax_per_batch = softmax_per_batch[-1] # This is the odd one out probability (at index k)
        softmax_per_batch = softmax_per_batch[0] # This is the similarity of object i,j (k) is the odd one out

        softmax_per_batch = softmax_per_batch.detach().cpu().numpy()
        softmax_all[:, i] = softmax_per_batch
        
        for s, softmax in enumerate(softmax_per_batch):
            if softmax < softmax_diff[s]:
                softmax_diff[s] = softmax
                most_important_dim[s] = i
        
    return softmax_all, softmax_diff, most_important_dim


def build_dataloader(triplet_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_dataset = build_triplet_dataset(triplet_path, device)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    return val_loader


def plot_bar(decisions, out_path="./"):

    dim = [f"Dim {str(i)}" for i in range(len(decisions))]
    
    df = pd.DataFrame({"softmax": decisions, "dim": dim} )
    fig = px.bar_polar(df, theta="dim", r="softmax", color="dim", template="simple_white")

    import plotly.graph_objects as go

    fig.update_layout(
    showlegend = False,
    polar = dict(
      bgcolor = "rgb(255, 255, 255)",
      angularaxis = dict(
        linewidth = 3,
        showline=False,
        showticklabels=False,
        # ticks='',
        linecolor='white'
      ),
      radialaxis = dict(
        side = "clockwise",
        showline = False,
        showticklabels = False,
        ticks = '',
        linewidth = 2,
        gridcolor = "white",
        gridwidth = 2,
      )
    ),
    paper_bgcolor = "rgb(255,255,255)"
    )

    for dim in fig.layout.annotations:
        dim.font.size = 20

    # fig.add_annotation(x=0.5, y=0.95, text="Dim 0", showarrow=False, font_size=15)







#     fig.add_trace(go.Scatter(
#     x=[0, 1, 2],
#     y=[3, 3, 3],
#     mode="lines+text",
#     name="Lines and Text",
#     text=["Text G", "Text H", "Text I"],
#     textposition="bottom center"
# ))
    # fig.update_traces(textposition='inside')
    # fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

    fig.show()


    fig.write_image(os.path.join(out_path, "bar_polar.png"), width=800, height=800)




def run_jackknife(human_weights, human_var, dnn_weights, dnn_var, image_filenames, triplet_path, plot_dir,  topk=12):
    """ This function runs the jackknife analysis for a given embedding"""

    # If all data on GPU, num workers need to be 0 and pin memory false
    device  = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
    val_loader = build_dataloader(triplet_path)

    softmax_human, indices = compute_softmax_decisions(human_weights, human_var, val_loader, device)
    softmax_dnn, indices = compute_softmax_decisions(dnn_weights, dnn_var, val_loader, device)

    # Indices human and indices DNN are the same
    most_diverging_indices = find_diverging_triplets(softmax_human, softmax_dnn, indices, topk)

    most_important_dim_human = dict()
    most_important_dim_dnn = dict()

    for key, value in most_diverging_indices.items():
        print(f"{key}: {value}")


        interesting_triplets = indices[value]
        softmax_jack_human, softmax_diff_human, important_human = jackknife(human_weights, human_var, interesting_triplets, device)
        most_important_dim_human[key] = important_human

        softmax_jack_dnn, softmax_diff_dnn, important_dnn = jackknife(dnn_weights, dnn_var, interesting_triplets, device)
        most_important_dim_dnn[key] = important_dnn


    i =0
    for key, value in most_diverging_indices.items():

        i += 1

        print(f"Starting {key}")
        
        dim_human = most_important_dim_human[key]
        dim_dnn = most_important_dim_dnn[key]

        interesting_triplets = indices[value]

        for i, (triplet, dim_h, dim_d) in enumerate(zip(interesting_triplets, dim_human, dim_dnn)):
            # fig, axes = plt.subplots(1, 3, figsize=(15, 5))


            fig = plt.figure(figsize=(15, 10))
            gs0 = gridspec.GridSpec(1, 3, figure=fig)
            gs0.update(top=0.95, bottom=0.7, left=0.25, right=0.75, wspace=0.0, hspace=0.0)



            for ctr, img_index in enumerate(triplet):
                ax = plt.subplot(gs0[ctr])

                img = Image.open(image_filenames[img_index])
                img = img.resize((224, 224))

                ax.imshow(img)
                ax.axis("off")


            plt.subplot(gs0[1]).set_title("Triplet", fontsize=20)

            # plt.tight_layout()
            # plt.savefig(os.path.join(plot_dir, "triplet_{}_{}.png".format(key, i)))
            # plt.close()
            

            human = human_weights[:, dim_h]
            human = np.argsort(-human)[:6]

            dnn = dnn_weights[:, dim_d]
            dnn = np.argsort(-dnn)[:6]

            if i==4:
                breakpoint()

            # fig = plt.figure(figsize=(15, 5))
            gs1 = gridspec.GridSpec(2, 3, figure=fig)
            gs1.update(top=0.65, bottom=0.1, left=0.05, right=0.45, wspace=0.0, hspace=0.0)
        
            for ctr, human_idx in enumerate(human):
                ax = plt.subplot(gs1[ctr])
                img = Image.open(image_filenames[human_idx])
                img = img.resize((224, 224), Image.Resampling.BILINEAR)
                ax.imshow(img)
                ax.axis("off")

            plt.subplot(gs1[1]).set_title("Dimension Human", fontsize=20)

            gs2 = gridspec.GridSpec(2, 3, figure=fig)
            gs2.update(top=0.65, bottom=0.1, left=0.55, right=0.95, wspace=0., hspace=0.0)
            
            for ctr, dnn_idx in enumerate(dnn):
                ax = plt.subplot(gs2[ctr])
                img = Image.open(image_filenames[dnn_idx])
                img = img.resize((224, 224))
                
                ax.imshow(img)
                ax.axis("off")

            plt.subplot(gs2[1]).set_title("Dimension DNN", fontsize=20)
            plt.savefig(os.path.join(plot_dir, "{}_{}.png".format(key, i)), dpi=300)
            plt.close()

