#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from turtle import color

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.transform import resize
import skimage.io as io

from deep_embeddings.utils.utils import load_sparse_codes, filter_embedding_by_behavior, rsm_pred, fill_diag, correlate_rsms, correlation_matrix
from scipy.stats import rankdata, pearsonr


parser = argparse.ArgumentParser(description='Compare human and DNN performance on the same task.')
parser.add_argument('--human_embedding_path', type=str, help='Path to human embedding matrix.')
parser.add_argument('--dnn_embedding_path', type=str, help='Path to DNN embedding matrix.')
parser.add_argument('--feature_path', type=str, help='Path to VGG feature matrix and filenames')

# TODO Check if reference images are loaded in line with the concept file (i.e. that path indices match?)

def load_inds_and_item_names(folder='./data/misc'):
    item_names = pd.read_csv(os.path.join(folder, 'item_names.tsv'), encoding='utf-8', sep='\t').uniqueID.values
    sortindex = pd.read_table(os.path.join(folder, 'sortindex'), header=None)[0].values
    
    return item_names, sortindex

def get_img_pairs(tril_indices, most_dissimilar):
    """ Returns the most dissimlar image pair indices in the reference images from the lower triangular RSM matrix"""
    tril_inds_i= tril_indices[0][most_dissimilar]
    tril_inds_j = tril_indices[1][most_dissimilar]
    img_pair_indices = np.array([(i,j) for i, j in zip(tril_inds_i, tril_inds_j)])

    return img_pair_indices

def load_concepts(folder='./data/misc'):
    concepts = pd.read_csv(os.path.join(folder, 'category_mat_manual.tsv'), encoding='utf-8', sep='\t')
    
    return concepts


def compare_modalities(weights_dnn, weights_human, duplicates=False):
    """ Compare the Human behavior embedding to the VGG embedding by correlating them! """
    assert weights_dnn.shape[0] == weights_human.shape[0], '\nNumber of items in weight matrices must align.\n'
    
    mod1_mod2_corrs = np.zeros(weights_dnn.shape[1])
    mod2_dims = []
    for d_mod1, w_mod1 in enumerate(weights_dnn.T):
        corrs = np.array([pearsonr(w_mod1, w_mod2) for w_mod2 in weights_human.T])
        if duplicates:
            mod2_dims.append(np.argmax(corrs))
        else:
            for d_mod2 in np.argsort(-corrs):
                if d_mod2 not in mod2_dims:
                    mod2_dims.append(d_mod2)
                    break
        mod1_mod2_corrs[d_mod1] = corrs[mod2_dims[-1]]
    mod1_dims_sorted = np.argsort(-mod1_mod2_corrs)
    mod2_dims_sorted = np.asarray(mod2_dims)[mod1_dims_sorted]
    corrs = mod1_mod2_corrs[mod1_dims_sorted]

    return mod1_dims_sorted, mod2_dims_sorted, corrs

def extract_concept(rsm, concept_assignments):
    pass

def plot_density_scatters(plots_dir, ref_images, rsm_mod1, rsm_mod2, concepts, top_k, mod1="Human", mod2="VGG16"):
    # TODO Replace RSM mod 1 and 2 with different names!
    assert rsm_mod1.shape == rsm_mod2.shape, '\nRSMs must be of equal size.\n'
    
    #create directory for density scatter plots
    scatter_dir = 'density_scatters'
    path = os.path.join(plots_dir, scatter_dir)
    if not os.path.exists(path):
        print(f'\nCreating directories...\n')
        os.makedirs(path)

    colors = ['silver', 'red', 'blue']

    for i, concept in enumerate(concepts.columns):
        
        # Find indices in the concept matrix where the concept is present
        concept_assignments = np.where(concepts.loc[:, concept] == 1)[0]

        # Extract the RSMs for the concept (i.e. the rows and columns corresponding to the concept)
        rsm_mod1_concept = rsm_mod1[concept_assignments]
        rsm_mod1_concept = rsm_mod1_concept[:, concept_assignments]
        rsm_mod2_concept = rsm_mod2[concept_assignments]
        rsm_mod2_concept = rsm_mod2_concept[:, concept_assignments]

        
        # Compute lower triangular parts of symmetric RSMs (zero elements above and including main diagonal)
        tril_inds = np.tril_indices(len(rsm_mod1_concept), k=-1)
        tril_mod1 = rsm_mod1_concept[tril_inds]
        tril_mod2 = rsm_mod2_concept[tril_inds]
        rho = pearsonr(tril_mod1, tril_mod2).round(5)

        # Find object pairs that are most dissimilar between modality one (i.e. behavior) and modality 2 (i.e. VGG 16)
        most_dissim_mod1 = np.argsort(rankdata(tril_mod1) - rankdata(tril_mod2))[::-1][:top_k]
        most_dissim_mod2 = np.argsort(rankdata(tril_mod2) - rankdata(tril_mod1))[::-1][:top_k]
        most_dissim_pairs = (most_dissim_mod1, most_dissim_mod2)

        # Store thes indices in a 1d array of all lower triangular indices
        categories = np.zeros(len(tril_mod1), dtype=int)
        categories[most_dissim_mod1] += 1
        categories[most_dissim_mod2] += 2

        # Store the lower triangular RSM values for both modalities along side the category cluster assignment for a single concept
        object_similarities = pd.DataFrame(np.c_[tril_mod1, tril_mod2, categories], columns=[mod1, mod2, 'category'])


        with sns.axes_style('white'):
            g = sns.jointplot(data=object_similarities, x=mod1, y=mod2, kind='scatter', 
                              hue='category', palette=dict(enumerate(colors)), legend=False, height=7)
            x = object_similarities[mod1]
            y = object_similarities[mod2]
            m, b = np.polyfit(x, y, deg=1)
            
            # Draw (regression) line of best fit
            g.ax_joint.plot(x, m*x+b, linewidth=2, c='black')
            g.ax_joint.set_xticks([])
            g.ax_joint.set_yticks([])
            g.ax_joint.set_xlabel(mod1, fontsize=13)
            g.ax_joint.set_ylabel(mod2, fontsize=13)

            # Annotate pvalue 
            pvalue_str = r'$\rho$' + ' = ' + str(rho)
            loc_x = np.min(tril_mod1)
            loc_y = np.max(tril_mod2)
            g.ax_joint.annotate(pvalue_str, (loc_x, loc_y), fontsize=11)
            plt.title(concept)

            # Save figure
            concept_path = os.path.join(path, "concepts")
            if not os.path.exists(concept_path):
                os.makedirs(concept_path)
            fname = os.path.join(concept_path, f'{concept}.jpg')
            g.savefig(fname, bbox_inches="tight")
            plt.close()

        # Visualize the most dissimilar object pairs as images!
        mods = [mod1, mod2]
        ref_images_c = ref_images[concept_assignments]
        for i, most_dissim in enumerate(most_dissim_pairs):
            img_pairs = get_img_pairs(tril_inds, most_dissim)
            plt.figure(figsize=(9, 6), dpi=300)
            ax = plt.subplot(111)

            for spine in ax.spines:
                ax.spines[spine].set_color(colors[i+1])
                ax.spines[spine].set_linewidth(1.75)

            pair_1, pair_2 = img_pairs[i]
            path_1 = ref_images_c[pair_1]
            path_2 = ref_images_c[pair_2]
            

            img_1 = resize(io.imread(path_1), (400, 400))
            img_2 = resize(io.imread(path_2), (400, 400))
            img = np.concatenate((img_1, img_2), axis=1)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(mods[i], fontsize=12)

            mod_str = mods[i].lower().split()
            mod_str = '_'.join(mod_str)
            dissim_path = os.path.join(path, "most_dissim_obj_pairs")
            if not os.path.exists(dissim_path):
                os.makedirs(dissim_path)
            plt.savefig(fname, bbox_inches="tight")
            plt.close()

def plot_most_dissim_pairs(plots_dir, ref_images, tril_mod1, tril_mod2, mod1, mod2, tril_inds, top_k):
    #create directory for density scatter plots
    PATH = os.path.join(plots_dir, ''.join(('density_scatters', '_', 'overall')))
    if not os.path.exists(PATH):
        print(f'\nCreating directories...\n')
        os.makedirs(PATH)

    # Compute pearson correlation of lower triangular parts of RSMs
    rho = pearsonr(tril_mod1, tril_mod2).round(3)

    # Subtract rank of each element in tril_mod1 from rank of each element in tril_mod2
    most_dissim_mod1 = np.argsort(tril_mod1 - tril_mod2)[::-1][:top_k]
    most_dissim_mod2 = np.argsort(tril_mod2 - tril_mod1)[::-1][:top_k]
    most_dissim_pairs = [most_dissim_mod1, most_dissim_mod2]
    colors = ['grey', 'red', 'blue']

    labels = np.zeros(len(tril_mod1))
    labels[most_dissim_mod1] += 1
    labels[most_dissim_mod2] += 2
    obj_sims = pd.DataFrame(np.c_[tril_mod1, tril_mod2, labels], columns=[mod1, mod2, 'labels'])
    with sns.axes_style('ticks'):
        g = sns.jointplot(data=obj_sims, x=mod1, y=mod2, hue='labels', palette=dict(enumerate(colors)), height=7, alpha=.6, kind='scatter', legend=False)
        x = obj_sims[mod1]
        y = obj_sims[mod2]
        m, b = np.polyfit(x, y, 1)
        g.ax_joint.plot(x, m*x+b, linewidth=2, c='black')
        g.ax_joint.set_xticks([])
        g.ax_joint.set_yticks([])
        g.ax_joint.set_xlabel(obj_sims.columns[0], fontsize=13)
        g.ax_joint.set_ylabel(obj_sims.columns[1], fontsize=13)
        g.ax_joint.annotate(''.join((r'$\rho$',' = ', str(rho))), (np.min(tril_mod1), np.max(tril_mod2)), fontsize=10)
        g.savefig(os.path.join(PATH, 'most_dissim_obj_pairs.jpg'))

    mods = [mod1, mod2]
    for i, most_dissim in enumerate(most_dissim_pairs):
        img_pairs = get_img_pairs(tril_inds, most_dissim)
        fig = plt.figure(figsize=(18, 12), dpi=100)
        ax = plt.subplot(111)

        for spine in ax.spines:
            ax.spines[spine].set_color(colors[i+1])
            ax.spines[spine].set_linewidth(1.75)

        ax.imshow(img_pairs)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(mods[i])
        plt.savefig(os.path.join(PATH, ''.join(('most_dissim_obj_pairs', '_', '_'.join(mods[i].lower().split()), '.jpg'))))
        plt.close()


def compare_human_dnn(human_embedding_path, dnn_embedding_path, feature_path):
    """Compare human and DNN performance on the same task."""

    weights_dnn = load_sparse_codes(dnn_embedding_path)
    weights_human = load_sparse_codes(human_embedding_path)

    filenames_path = os.path.join(feature_path, 'file_names.txt')
    if not os.path.exists(filenames_path):
        raise FileNotFoundError("File names not found in DNN activation path {}".format(feature_path))

    filenames = np.loadtxt(filenames_path, dtype=str)

    # Filter out images without behavioral data
    weights_dnn, ref_images = filter_embedding_by_behavior(weights_dnn, filenames)
    # ref_images = load_ref_images(ref_images)

    # Get rsm matrices
    rsm_dnn = correlation_matrix(weights_dnn)
    rsm_human = correlation_matrix(weights_human)

    rho = correlate_rsms(rsm_dnn, rsm_human, "correlation")
    corr = pearsonr(rsm_dnn.flatten(), rsm_human.flatten())[0].round(3)
    print("Correlation between human and DNN embeddings: {}".format(rho))

    # Whether to perform mind-machine comparison with VGG 16 dimensions that allow for duplicate or unique latent dimensions
    duplicates = False
    behavior_dims, dnn_dims, mind_machine_corrs  = compare_modalities(weights_human, weights_dnn, duplicates=duplicates)

    # item_names, sortindex = load_inds_and_item_names()
    concepts = load_concepts()

    plot_dir_comparison = os.path.dirname(os.path.dirname(dnn_embedding_path))
    plot_dir_comparison = os.path.join(plot_dir_comparison, "analyses", "human_dnn")
    if not os.path.exists(plot_dir_comparison):
        os.makedirs(plot_dir_comparison)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(range(len(mind_machine_corrs)), mind_machine_corrs, color='black', linewidth=2)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Correlation between human and DNN embeddings")
    fig.savefig(os.path.join(plot_dir_comparison, "human_dnn_dimension_correlation.jpg"))

        
    plot_density_scatters(plots_dir=plot_dir_comparison, ref_images=ref_images, rsm_mod1=rsm_human, rsm_mod2=rsm_dnn, 
                          mod1='Human Behavior', mod2='VGG 16', concepts=concepts, top_k=20)
    plot_most_dissim_pairs(plots_dir=plot_dir_comparison, ref_images=ref_images, tril_mod1=tril_human,
                           tril_mod2=tril_dnn, mod1='Human Behavior', mod2='VGG 16', tril_inds=tril_inds, top_k=20)


if __name__ == '__main__':
    args = parser.parse_args()
    compare_human_dnn(args.human_embedding_path, args.dnn_embedding_path, args.feature_path)

