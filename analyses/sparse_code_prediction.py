#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import joblib
import multiprocessing
import os
import random
import time
import deep_embeddings.utils as utils
import numpy as np

from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from typing import Tuple,  Any

os.environ['OMP_NUM_THREADS']='1'

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--dnn_path', type=str,
        help='PATH to DNN object representations')
    aa('--embedding_path', type=str,
        help='PATH to Embedding object representations')
    aa('--num_workers', type=int, default=multiprocessing.cpu_count()-1,
        help='number of processes to run in parallel')
    aa('--k_folds', type=int, default=10,
        choices=[3, 5, 10, 20],
        help='number of folds in cross-validation')
    aa('--rnd_seed', type=int, default=42)
    args = parser.parse_args()
    return args

args = parseargs()
global num_workers
num_workers = args.num_workers
print(f'Initialized {num_workers} workers to run task in parallel.\n')
time.sleep(1.0)

def load_activations(PATH:str) -> np.ndarray:
    if PATH.endswith('npy'):
        with open(PATH, 'rb') as f:
            F = np.load(f)
    else:
        F = np.loadtxt(PATH)
    return F

def center_activations(F:np.ndarray) -> np.ndarray:
    return F - F.mean(axis=0)

def kfold_cv(X:np.ndarray, y:np.ndarray, lmbda:float, rnd_seed:int, k_folds:int) -> Tuple:
    kf = KFold(n_splits=k_folds, random_state=rnd_seed, shuffle=True)
    #initialize an l1 and l2 regularized regression model (i.e., Elastic net = Tikhonov regression with an additional l1 penalty)
    reg_model = ElasticNet(alpha=1.0, max_iter=5000, l1_ratio=lmbda, random_state=rnd_seed)
    r2_scores = np.zeros(k_folds)
    for k, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        reg_model.fit(X_train, y_train)
        y_pred = reg_model.predict(X_test)
        r2_scores[k] = r2_score(y_test, y_pred)
    r2 = np.mean(r2_scores)
    return r2, reg_model

def train_test_split(X:np.ndarray, y:np.ndarray, train_frac:float=.9, shuffle:bool=True) -> Tuple[np.ndarray]:
    """split the data into train and test sets"""
    N = X.shape[0]
    if shuffle:
        rnd_perm = np.random.permutation(N)
        train_indices = rnd_perm[:int(N*train_frac)]
        test_indices = rnd_perm[int(N*train_frac):]
    else:
        train_indices = np.arange(0, int(N*train_frac))
        test_indices = np.arange(int(N*train_frac), N)
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

def train_and_test(X:np.ndarray, y:np.ndarray, lmbda:float, rnd_seed:int) -> Tuple[float, Any]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_frac=.9, shuffle=True)
    #initialize an l1 and l2 regularized regression model (i.e., Elastic net = Tikhonov regression with an additional l1 penalty)
    reg_model = ElasticNet(alpha=1.0, max_iter=5000, l1_ratio=lmbda, random_state=rnd_seed)
    reg_model.fit(X_train, y_train)
    y_pred = reg_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2, reg_model

def grid_search_cv(i:int, X:np.ndarray, y:np.ndarray, lambdas:np.ndarray, rnd_seed:int, k_folds:int) -> Tuple[int, float, Any]:
    print(f"\n...Started grid search for latent dimension: {i}.")
    results = np.array([kfold_cv(X=X, y=y, lmbda=lmbda, rnd_seed=rnd_seed, k_folds=k_folds)[0] for lmbda in lambdas])
    print(f"...Finished grid search for latent dimension: {i}.")
    print("...Now performing model fit with final lambda value.")
    best_lmbda = lambdas[np.argmax(results)]
    r2, reg_model = kfold_cv(X=X, y=y, lmbda=best_lmbda, rnd_seed=rnd_seed, k_folds=k_folds)
    time.sleep(5.0)
    print(f"...Finished optimization for latent dimension: {i}.\n")
    return (i, r2, reg_model)

def prediction(X:np.ndarray, Y:np.ndarray, lambdas:np.ndarray, results_path:str, rnd_seed:int, k_folds:int) -> np.ndarray:
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.starmap_async(grid_search_cv, [(i, X, y, lambdas, rnd_seed, k_folds) for i, y in enumerate(Y.T)]).get()
        pool.close()
        pool.join()
    r2_scores = np.zeros(len(results))
    for (dim, r2, predictor) in results:
        joblib.dump(predictor, os.path.join(results_path, f'predictor_{dim:02d}.joblib'))
        r2_scores[dim] = r2
    return r2_scores

def run(dnn_path:str, spose_path:str, k_folds:int, rnd_seed:int) -> None:
    F = load_activations(dnn_path)
    X = center_activations(F)
    Y, _ = utils.load_sparse_codes(spose_path)

    # NOTE remove all negative values!
    X = np.maximum(X, 0)
    Y = np.maximum(Y, 0)

    Y = Y[:,0].reshape(-1,1)

    print(f'\nShape of SPoSE embedding matrix: {Y.shape}')
    print(f'Shape of DNN feature matrix: {X.shape}\n')


    assert X.shape[0] == Y.shape[0], '\nNumber of objects in SPoSE embedding and DNN feature matrix must be the same.\n'

    # results_path = os.path.join(spose_path, 'sparse_code_predictions')
    results_path = './sparse_code_predictions'
    if not os.path.exists(results_path):
        print('\n...Creating directories.\n')
        os.makedirs(results_path)

    r2_scores = prediction(X=X, Y=Y, lambdas=np.arange(0, 1.2, 1.2), results_path=results_path, rnd_seed=rnd_seed, k_folds=k_folds)

    print(r2_scores)

    with open(os.path.join(results_path, 'r2_scores.npy'), 'wb') as f:
        np.save(f, r2_scores)

if __name__ == "__main__":
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)

    # NOTE DNN oath is the path to the DNN activations for all objects.
    # i.e. /LOCAL/fmahner/THINGS/vgg_features6
    # NOTE Spose_path = VICE_path. Loads weights from an embedding in the form of a text file. 
    # Need to store VICE embedding as text file for this.
    # NOTE k_folds = 5
    # NOTE rnd_seed = 42
    
    args.dnn_path = '/home/florian/THINGS/vgg_bn_features_behavior/features.npy'
    args.embedding_path = "../learned_embeddings/weights_things_behavior_8196bs_adaptive_half/params/pruned_q_mu_epoch_5000.txt"
    args.k_folds = 2
    args.rnd_seed = 42

    run(
        dnn_path=args.dnn_path,
        spose_path=args.embedding_path,
        k_folds=args.k_folds,
        rnd_seed=args.rnd_seed,
        )
