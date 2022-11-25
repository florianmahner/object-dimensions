#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default="./results/gamma_analysis")
parser.add_argument("--key", type=str, default="gamma", help="Key to aggregate over that is contained in the path")

def map_path_to_param(path):
    split = path.split("/")
    hparams = dict()
    for i, key in enumerate(["iden", "model", "n_samples", "modality", "prior",
                "init_dim", "batch_size", "gamma", "seed"]):
        hparams[key] = split[i+2]

    return hparams
            
def aggregate_params(base_dir, key):
    """ Aggregate all the params for a given key, i.e. seeds or different gamma values """
    params = {}
    file_list = glob(os.path.join(base_dir + "/**/parameters.npz"), recursive=True)
    hparams = map_path_to_param(file_list[0])
    key_val = hparams[key]

    for file in file_list:
        params[key_val] = np.load(file)

    return params

def aggregate_results(params):
    """ We find the best model for our combined results by evaluating the validation loss """
    for key, val in params.items():
        val_loss = val["val_loss"]
        best_val_loss = min(val_loss)
        print("Gamma {} has best val loss of {}".format(key, best_val_loss))


if __name__ == '__main__':
    args = parser.parse_args()
    params = aggregate_params(args.base_dir, args.key)
    aggregate_results(params)
