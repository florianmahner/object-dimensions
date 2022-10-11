#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import joblib
import multiprocessing
import os
import random

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from deep_embeddings.utils import utils

from sklearn.linear_model import RidgeCV

os.environ['OMP_NUM_THREADS'] = '32'

parser = argparse.ArgumentParser(description='Run ridge regression on DNN features and embedding matrix.')
parser.add_argument("--dnn_path", type=str, default="./data/vgg_bn_features_12/features.npy", help='Path to DNN features.')
parser.add_argument("--embedding_path", type=str, default="./embedding/weights/params/pruned_params_epoch_1000.txt", help='Path to embedding matrix.')
parser.add_argument("--k_folds", type=int, default=4, help='Number of folds for cross-validation.')
parser.add_argument("--rnd_seed", type=int, default=42, help='Random seed for cross-validation.')


def plot_predictions(r2_scores, results_path):
    fig, ax  = plt.subplots(1)
    sns.lineplot(x=range(len(r2_scores)), y=r2_scores, ax=ax)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('R2 score')
    fig.tight_layout()
    out_path = os.path.join(results_path, "r2_scores.png")
    plt.savefig(out_path, dpi=300)

def run_ridge_regression(dnn_path, embedding_path, k_folds):
    global num_workers
    num_workers = multiprocessing.cpu_count()-1
    print(f'Initialized {num_workers} workers to run task in parallel.\n')

    base_path = os.path.dirname(os.path.dirname(embedding_path))
    results_path = os.path.join(base_path, "analyses", "sparse_codes")

    if not os.path.exists(results_path):
        print('\n...Creating directories.\n')
        os.makedirs(results_path)

    X = utils.load_deepnet_activations(dnn_path)
    Y = utils.load_sparse_codes(embedding_path)

    print(f'\nShape of embedding matrix: {Y.shape}')
    print(f'Shape of DNN feature matrix: {X.shape}\n')

    assert X.shape[0] == Y.shape[0], '\nNumber of objects in embedding and DNN feature matrix must be the same.\n'

    r2_scores = []
    alphas = []
    for dim, y in enumerate(Y.T):
        # ridge_cv = RidgeCV(scoring='r2', cv=k_folds, alphas=np.arange(0.4, 2.0, 0.2))
        ridge_cv = RidgeCV(scoring='r2', cv=k_folds, alphas=[1.0])
        y = y.reshape(-1,1)
        ridge_cv.fit(X, y)
        r2 = ridge_cv.score(X, y)
        r2_scores.append(r2)
        alpha = ridge_cv.alpha_
        alphas.append(alpha)
        print(f'R2score {r2}, Best alpha: {alpha}')
        joblib.dump(ridge_cv, os.path.join(results_pajth, f'predictor_{dim:02d}.joblib'))

    with open(os.path.join(results_path, 'r2_scores.npy'), 'wb') as f:
        np.save(f, r2_scores)

    plot_predictions(r2_scores, results_path)
    

if __name__ == "__main__":
    args = parser.parse_args()

    # args.dnn_path = '/LOCAL/fmahner/THINGS/vgg_bn_features_12/features.npy'
    # args.embedding_path = '/LOCAL/fmahner/DeepEmbeddings/learned_embeddings/weights_vgg_12_512bs/params/pruned_q_mu_epoch_500.txt'
    # args.k_folds = 4
    # args.rnd_seed = 42

    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    run_ridge_regression(dnn_path=args.dnn_path, embedding_path=args.embedding_path, k_folds=args.k_folds)
