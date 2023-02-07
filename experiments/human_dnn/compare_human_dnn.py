#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" This is the main script for the human vs DNN comparison """

import os
from glob import glob
import toml

import numpy as np
from deep_embeddings.utils.utils import load_sparse_codes, load_image_data, create_path_from_params, compute_rdm, fill_diag
from experiments.human_dnn.jackknife import run_jackknife
from experiments.human_dnn.embedding_analysis import run_embedding_analysis
from deep_embeddings import ExperimentParser
from scipy.stats import pearsonr, rankdata


parser = ExperimentParser(description='Analysis and comparison of embeddings')
parser.add_argument("--dnn_path", type=str, help="Path to the base directory of the experiment runs")
parser.add_argument("--human_path_base", type=str, help="Path to the base directory of the human embedding")
parser.add_argument("--human_path_comp", type=str, help="Path to the base directory of the human embedding")
parser.add_argument("--img_root", type=str, help="Path to the image root directory used for training the DNN. Contains behavior images and plus")
parser.add_argument("--triplet_path", type=str, help="Path to the behavior triplets used for jackknife analysis")
parser.add_argument("--evaluation_key", type=str, help="Key to aggregate over that is contained in the path")
parser.add_argument("--analysis_key", type=str, help="Key to analyze that is contained in the parameters.npz file")


def aggregate_files(base_dir, eval_key="seed"):
    """ Aggregate all the files for a given key, i.e. seeds or different beta values """
    file_list = glob(os.path.join(base_dir + "/**/parameters.npz"), recursive=True)
    print("Number of confiugrations found: {}".format(len(file_list)))
    if not file_list:
        raise ValueError("No parameters.npz files found in {}".format(base_dir))

    param_dict = {}
    for file in file_list:
        config_path = os.path.join(os.path.dirname(os.path.dirname(file)), "config.toml")
        config = toml.load(config_path)
        try:
            key_val = config[eval_key]
        except KeyError:
            print("Key {} not found in config file {}".format(eval_key, config_path))
            continue
        param_dict[key_val] = (np.load(file), file)

    if len(param_dict.keys()) != len(file_list):
        print("""Number of configurations found do not match the number of paramers.npz. Maybe the key
            {} is not a hyperparameter?""".format(eval_key))

    return param_dict


def aggregate_values(file_dict, analysis_key="val_loss", mode="min"):
    """ Aggregate the values of a given key over all the files in the dictionary """
    best_value = float("inf")
    best_file = None
    best_key = ""
    best_file_path = ""

    eval_fn = eval(mode)

    for key, (val, file_path) in file_dict.items():
        value = val[analysis_key]
        value = eval_fn(value)

        print(key, value)

        if value < best_value:
            best_value = value
            best_file = val
            best_key = key
            best_file_path = file_path

    print("Best {} for analysis is {} with value {}".format(analysis_key, best_key, best_value))
    
    return best_file, best_file_path


def find_best_embedding(path, evaluation_key="seed", analysis_key="val_loss"):
    """ We aggregate across different models in the same directory and then try to correlate """
    files = aggregate_files(path, evaluation_key)
    params, fpath = aggregate_values(files, analysis_key)
    return params, fpath


def kmeans(mat, K=50):
    ''' write a function that does k means clustering on an image '''
    # convert to np.float32
    Z = mat.reshape(-1, 1)
    Z = np.float32(Z)
    
    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center= cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((mat.shape))
    return res2

def kmeans_sklearn(img, k):
    """ Write a function that segments a 2d grayscale image using kmeans clustering and sklearn """
    from sklearn.cluster import KMeans
    img_flat = img.reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_flat)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    img_kmeans = centers[labels].reshape(img.shape)
    return img_kmeans


def get_rdm(embedding, method="correlation"):
    embedding = np.maximum(0, embedding)
    rsm = compute_rdm(embedding, method)
    rsm = fill_diag(rsm)
    rsm = rankdata(rsm).reshape(rsm.shape)

    return rsm

def normalise_rdm(rdm):
    rdm = rdm / np.max(rdm)
    return rdm


def compare_human_dnn(args):
    # TODO The RSM Plots
    # TODO The plot comparison with SPoSE
    
    # Check if DNN path is base path points to a trained parameter file
    if os.path.isfile(args.dnn_path):
        dnn_embedding, dnn_var = load_sparse_codes(args.dnn_path, with_var=True)    
    else:
        dnn_params, fpath = find_best_embedding(args.dnn_path, args.evaluation_key, args.analysis_key)
        args.dnn_path = fpath
        dnn_embedding, dnn_var = load_sparse_codes(dnn_params, with_var=True)

    if os.path.isfile(args.human_path_base):
        human_embedding, human_var = load_sparse_codes(args.human_path_base, with_var=True)
    else:
        human_params, fpath = find_best_embedding(args.human_path_base, args.evaluation_key, args.analysis_key)
        human_embedding, human_var = load_sparse_codes(human_params, with_var=True)


    if os.path.isfile(args.human_path_comp):
        human_embedding_comp, human_var_comp = load_sparse_codes(args.human_path_comp, with_var=True)


    


    # Load the image data
    plot_dir = create_path_from_params(args.dnn_path, "analyses", "human_dnn")
    print("Save all human dnn comparisons to {}".format(plot_dir))


    image_filenames, indices = load_image_data(args.img_root, filter_behavior=True)

    # TODO Maybe make an additional assert statement to check if the number of images in the embedding match the number of loaded images
    
    dnn_embedding = dnn_embedding[indices]
    dnn_var = dnn_var[indices]


    method = 'correlation'

    rsm_1 = get_rdm(dnn_embedding, method)
    rsm_2 = get_rdm(human_embedding, method)



    tril_inds = np.tril_indices(len(rsm_1), k=-1)
    tril_1 = rsm_1[tril_inds]
    tril_2 = rsm_2[tril_inds]
    rho = pearsonr(tril_1, tril_2)[0].round(3)
    print(f'\nCorrelation between RSMs: {rho:.3f}\n')


    rsm_1_km = kmeans(rsm_1, 20)
    rsm_2_km = kmeans(rsm_2, 20)

    # rsm_1_km = normalise_rdm(rsm_1_km)
    # rsm_2_km = normalise_rdm(rsm_1_km)

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))


    ax1.imshow(rsm_1_km, cmap='viridis_r')
    ax1.set_title("Behavior")
    ax2.imshow(rsm_2_km, cmap='viridis_r')
    ax2.set_title("Deep CNN")

    plt.savefig("rdm_test.png", dpi=300)




    # run_embedding_analysis(human_embedding, dnn_embedding, image_filenames, plot_dir)

    # plot_dir = create_path_from_params(args.dnn_path, "analyses", "jackknife_human_dnn")
    # run_jackknife(human_embedding, human_var, dnn_embedding, dnn_var,  image_filenames, args.triplet_path, plot_dir)

    # if args.human_path_comp:
    #     plot_dir = create_path_from_params(args.dnn_path, "analyses", "jacknife_human_human")
    #     run_jackknife(human_embedding, human_var, human_embedding_comp, human_var_comp, image_filenames, args.triplet_path, plot_dir, 12, "human")


if __name__ == '__main__':
    args = parser.parse_args()
    compare_human_dnn(args)