#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.transform import resize
import skimage.io as io
from yaml import load


from deep_embeddings.utils.utils import load_sparse_codes, filter_embedding_by_behavior, rsm_pred, fill_diag, correlate_rsms, correlation_matrix
from scipy.stats import rankdata

parser = argparse.ArgumentParser(description='Compare human and DNN performance on the same task.')
parser.add_argument('--human_embedding_path', type=str, help='Path to human embedding matrix.')
parser.add_argument('--dnn_embedding_path', type=str, help='Path to DNN embedding matrix.')
parser.add_argument('--feature_path', type=str, help='Path to VGG feature matrix and filenames')

def pearsonr(u, v, a_min=-1., a_max=1.):
    u_c = u - np.mean(u)
    v_c = v - np.mean(v)
    num = u_c @ v_c
    denom = np.linalg.norm(u_c) * np.linalg.norm(v_c)
    rho = (num / denom).clip(min=a_min, max=a_max)

    return rho

def compare_modalities(weights_dnn, weights_human, duplicates=False):
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

def get_img_pairs(tril_inds, most_dissim, ref_images):
    tril_inds_i = tril_inds[0][most_dissim]
    tril_inds_j = tril_inds[1][most_dissim]
    # ref_images_i = ref_images[tril_inds_i]
    # ref_images_j = ref_images[tril_inds_j]

    img_pair_indices = np.array([(i,j) for i, j in zip(tril_inds_i, tril_inds_j)])
    # img_pair_indices = np.concatenate(tril_inds_i, tril_inds_j)
    # img_pairs = np.concatenate((np.concatenate(ref_images_i, axis=1), np.concatenate(ref_images_j, axis=1)), axis=0)

    return img_pair_indices

def load_concepts(folder='./data/misc'):
    concepts = pd.read_csv(os.path.join(folder, 'category_mat_manual.tsv'), encoding='utf-8', sep='\t')
    return concepts


def load_ref_images(file_paths, compress=True):
    ref_images = []
    if compress:
        for file in file_paths:
            image = resize(io.imread(file), (224, 224), anti_aliasing=True)
            ref_images.append(image)
    ref_images = np.array(ref_images)
    return ref_images


def plot_density_scatters(plots_dir, ref_images, rsm_mod1, rsm_mod2, mod1, mod2, concepts, top_k):
    #create directory for density scatter plots
    scatter_dir = 'density_scatters'
    PATH = os.path.join(plots_dir, scatter_dir)
    if not os.path.exists(PATH):
        print(f'\nCreating directories...\n')
        os.makedirs(PATH)

    colors = ['silver', 'red', 'blue']

    for i, concept in enumerate(concepts.columns):
        assignments = np.where(concepts.loc[:, concept] == 1)[0]
        #create RSMs for each modality wrt current concept
        rsm_mod1_c = rsm_mod1[assignments]
        rsm_mod1_c = rsm_mod1_c[:, assignments]
        rsm_mod2_c = rsm_mod2[assignments]
        rsm_mod2_c = rsm_mod2_c[:, assignments]
        assert rsm_mod1_c.shape == rsm_mod2_c.shape, '\nRSMs must be of equal size.\n'
        #compute lower triangular parts of symmetric RSMs (zero elements above and including main diagonal)
        tril_inds = np.tril_indices(len(rsm_mod1_c), k=-1)
        tril_mod1 = rsm_mod1_c[tril_inds]
        tril_mod2 = rsm_mod2_c[tril_inds]
        #calculate Pearson correlation between lower triangular parts of modality specific RSMs
        rho = pearsonr(tril_mod1, tril_mod2).round(3)

        #find object pairs that are most dissimilar between minds (SPoSE Behavior) and machines (SPoSE VGG 16)
        most_dissim_mod1 = np.argsort(rankdata(tril_mod1) - rankdata(tril_mod2))[::-1][:top_k]
        most_dissim_mod2 = np.argsort(rankdata(tril_mod2) - rankdata(tril_mod1))[::-1][:top_k]
        #most_dissim_mod1 = np.argsort(tril_mod1 - tril_mod2)[::-1][:top_k]
        #most_dissim_mod2 = np.argsort(tril_mod2 - tril_mod1)[::-1][:top_k]
        most_dissim_pairs = [most_dissim_mod1, most_dissim_mod2]

        categories = np.zeros(len(tril_mod1), dtype=int)
        categories[most_dissim_mod1] += 1
        categories[most_dissim_mod2] += 2

        obj_sims = pd.DataFrame(np.c_[tril_mod1, tril_mod2, categories], columns=[mod1, mod2, 'cat'])
        with sns.axes_style('white'):
            g = sns.jointplot(data=obj_sims, x=mod1, y=mod2, kind='scatter', hue='cat', palette=dict(enumerate(colors)), legend=False, height=7)
            x = obj_sims[mod1]
            y = obj_sims[mod2]
            m, b = np.polyfit(x, y, 1)
            #draw (regression) line of best fit
            g.ax_joint.plot(x, m*x+b, linewidth=2, c='black')
            g.ax_joint.set_xticks([])
            g.ax_joint.set_yticks([])
            g.ax_joint.set_xlabel(mod1, fontsize=13)
            g.ax_joint.set_ylabel(mod2, fontsize=13)
            g.ax_joint.annotate(''.join((r'$\rho$',' = ', str(rho))), (np.min(tril_mod1), np.max(tril_mod2)), fontsize=11)
            #plt.title(concept)
            g.savefig(os.path.join(PATH, ''.join((concept, '.jpg'))))
            plt.close()

        mods = [mod1, mod2]
        ref_images_c = ref_images[assignments]
        for i, most_dissim in enumerate(most_dissim_pairs):
            img_pairs = get_img_pairs(tril_inds, most_dissim, ref_images_c)
            fig = plt.figure(figsize=(18, 12), dpi=100)
            ax = plt.subplot(111)

            for spine in ax.spines:
                ax.spines[spine].set_color(colors[i+1])
                ax.spines[spine].set_linewidth(1.75)

            breakpoint()
            ax.imshow(img_pairs)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(mods[i], fontsize=12)
            plt.savefig(os.path.join(PATH, ''.join(('most_dissim_obj_pairs', '_', concept, '_', '_'.join(mods[i].lower().split()), '.jpg'))))
            plt.close()

def plot_most_dissim_pairs(plots_dir, ref_images, tril_mod1, tril_mod2, mod1, mod2, tril_inds, top_k):
    #create directory for density scatter plots
    PATH = os.path.join(plots_dir, ''.join(('density_scatters', '_', 'overall')))
    if not os.path.exists(PATH):
        print(f'\nCreating directories...\n')
        os.makedirs(PATH)

    rho = pearsonr(tril_mod1, tril_mod2).round(3)
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
        img_pairs = get_img_pairs(tril_inds, most_dissim, ref_images)
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

def load_inds_and_item_names(folder='./data/misc'):
    item_names = pd.read_csv(os.path.join(folder, 'item_names.tsv'), encoding='utf-8', sep='\t').uniqueID.values
    sortindex = pd.read_table(os.path.join(folder, 'sortindex'), header=None)[0].values
    
    return item_names, sortindex


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
    rsm_dnn = fill_diag(correlation_matrix(weights_dnn))  
    rsm_human = fill_diag(correlation_matrix(weights_human))

    # rsm_dnn = fill_diag(rsm_pred(weights_dnn))
    # rsm_human = fill_diag(rsm_pred(weights_human))

    # Correlate embeddings
    tril_inds = np.tril_indices(len(rsm_human), k=-1)
    tril_human = rsm_human[tril_inds]
    tril_dnn = rsm_dnn[tril_inds]
    rho = pearsonr(tril_human, tril_dnn).round(3)
    print("Correlation between human and DNN embeddings: {}".format(rho))
    

    # Whether to perform mind-machine comparison with VGG 16 dimensions that allow for duplicate or unique latent dimensions
    duplicates = False
    behavior_dims, dnn_dims, mind_machine_corrs  = compare_modalities(weights_human, weights_dnn, duplicates=duplicates)


    item_names, sortindex = load_inds_and_item_names()
    concepts = load_concepts()

    plot_dir_comparison = os.path.dirname(os.path.dirname(dnn_embedding_path))
    plot_dir_comparison = os.path.join(plot_dir_comparison, "analyses", "huamn_dnn")
    if not os.path.exists(plot_dir_comparison):
        os.makedirs(plot_dir_comparison)

        
    plot_density_scatters(plots_dir=plot_dir_comparison, ref_images=ref_images, rsm_mod1=rsm_human, rsm_mod2=rsm_dnn, 
                          mod1='Human Behavior', mod2='VGG 16', concepts=concepts, top_k=20)
    plot_most_dissim_pairs(plots_dir=plot_dir_comparison, ref_images=ref_images, tril_mod1=tril_human,
                           tril_mod2=tril_dnn, mod1='Human Behavior', mod2='VGG 16', tril_inds=tril_inds, top_k=20)

    


if __name__ == '__main__':
    args = parser.parse_args()
    compare_human_dnn(args.human_embedding_path, args.dnn_embedding_path, args.feature_path)

