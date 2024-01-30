#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default="./results/hyperparam_analysis")
parser.add_argument(
    "--key",
    type=str,
    default="",
    help="Key to aggregate over that is contained in the path",
)


def map_path_to_param(path):
    """Currently wrong. Store the log path format somewhere!"""
    split = path.split("/")
    hparams = dict()
    for i, key in enumerate(
        [
            "iden",
            "model",
            "n_samples",
            "modality",
            "prior",
            "init_dim",
            "batch_size",
            "spike",
            "slab",
            "pi",
            "gamma",
            "seed",
        ]
    ):
        hparams[key] = split[i + 2]

    return hparams


def aggregate_params(base_dir, key=None):
    """Aggregate all the params for a given key, i.e. seeds or different gamma values"""

    file_list = glob(os.path.join(base_dir + "/**/parameters.npz"), recursive=True)
    print("Number of confiugrations found: {}".format(len(file_list)))
    if not file_list:
        raise ValueError("No parameters.npz files found in {}".format(base_dir))

    if key:
        params = {}
        hparams = map_path_to_param(file_list[0])
        key_val = hparams[key]

        for file in file_list:
            params[key_val] = np.load(file)

    else:
        best_val_loss = float("inf")
        best_file = None
        best_params = {}
        for file in file_list:

            params_file = np.load(file)
            val_loss = params_file["val_loss"].min()

            print(val_loss, "niter", len(params_file["dim_over_time"]))

            if val_loss < best_val_loss:
                best_params = params_file
                best_val_loss = val_loss
                best_file = file

        print("Best file is {}".format(best_file))

    return best_params


def aggregate_results(params):
    """We find the best model for our combined results by evaluating the validation loss"""
    for key, val in params.items():
        val_loss = val["val_loss"]
        best_val_loss = min(val_loss)
        print("Gamma {} has best val loss of {}".format(key, best_val_loss))


if __name__ == "__main__":
    args = parser.parse_args()
    params = aggregate_params(args.base_dir, args.key)
    # aggregate_results(params)
