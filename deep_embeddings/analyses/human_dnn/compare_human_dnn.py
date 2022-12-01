#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" This is the main script for the human vs DNN comparison """

import os
from glob import glob
import toml

import numpy as np
from deep_embeddings.utils.utils import load_sparse_codes, load_image_data, create_path_from_params
from deep_embeddings.analyses.human_dnn.jackknife import run_jackknife
from deep_embeddings.analyses.human_dnn.embedding_analysis import run_embedding_analysis
from deep_embeddings import ExperimentParser


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

    if os.path.isfile(args.human_path_comp):
        human_embedding_comp, human_var_comp = load_sparse_codes(args.human_path_comp, with_var=True)
    
    else:
        human_params, fpath = find_best_embedding(args.human_path_base, args.evaluation_key, args.analysis_key)
        human_embedding, human_var = load_sparse_codes(human_params, with_var=True)



    # Load the image data
    plot_dir = create_path_from_params(args.dnn_path, "analyses", "human_dnn")
    print("Save all human dnn comparisons to {}".format(plot_dir))


    image_filenames, indices = load_image_data(args.img_root, filter_behavior=True)

    # TODO Maybe make an additional assert statement to check if the number of images in the embedding match the number of loaded images
    
    dnn_embedding = dnn_embedding[indices]
    dnn_var = dnn_var[indices]

    # run_embedding_analysis(human_embedding, dnn_embedding, image_filenames, plot_dir)

    plot_dir = create_path_from_params(args.dnn_path, "analyses", "jackknife_human_dnn")
    run_jackknife(human_embedding, human_var, dnn_embedding, dnn_var,  image_filenames, args.triplet_path, plot_dir)

    if args.human_path_comp:
        plot_dir = create_path_from_params(args.dnn_path, "analyses", "jacknife_human_human")
        run_jackknife(human_embedding, human_var, human_embedding_comp, human_var_comp, image_filenames, args.triplet_path, plot_dir)


if __name__ == '__main__':
    args = parser.parse_args()
    compare_human_dnn(args)