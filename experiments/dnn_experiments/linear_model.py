#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import joblib
import multiprocessing
import os
import random

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from object_dimensions.utils import (
    load_deepnet_activations,
    load_sparse_codes,
)

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score

from tomlparse import argparse


os.environ["OMP_NUM_THREADS"] = "32"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ridge regression on DNN features and embedding matrix."
    )
    parser.add_argument(
        "--dnn_path",
        type=str,
        default="./data/vgg_bn_features_12",
        help="Path to DNN features.",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        default="./embedding/weights/params/pruned_params_epoch_1000.txt",
        help="Path to embedding matrix.",
    )
    parser.add_argument(
        "--k_folds", type=int, default=4, help="Number of folds for cross-validation."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for cross-validation."
    )
    return parser.parse_args()


def plot_predictions(r2_scores, results_path):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    sns.set_style("white")
    sns.set_context("paper", font_scale=1.5)
    sns.lineplot(x=range(len(r2_scores)), y=r2_scores, ax=ax, color="black")
    ax.set_xlabel("Dimension")
    ax.set_ylabel(r"$R^2$ score")
    # ax.set_title("DNN regression on embedding matrix.")
    fig.tight_layout()

    out_path = os.path.join(
        results_path, "r2_scores.pdf", dpi=300, bbox_inches="tight", pad_inches=0
    )
    plt.savefig(out_path, dpi=300)


def run_ridge_regression(dnn_path, embedding_path, k_folds):
    global num_workers
    num_workers = multiprocessing.cpu_count() - 1
    print(f"Initialized {num_workers} workers to run task in parallel.\n")

    base_path = os.path.dirname(os.path.dirname(embedding_path))
    results_path = os.path.join(base_path, "analyses", "sparse_codes")

    if not os.path.exists(results_path):
        print("\n...Creating directories.\n")
        os.makedirs(results_path)

    # Check if we want to delete and wait for input
    if os.listdir(results_path):
        print(
            f"\n\n...Directory {results_path} is not empty. Do you want to delete all previous results? [y/n]"
        )
        user_input = input()
        if user_input == "y":
            # Delete all previous sparse codes in the dir
            print(
                f"\n...Deleting all previous results in {results_path} to start fresh.\n"
            )
            for file in os.listdir(results_path):
                if file.endswith(".joblib"):
                    os.remove(os.path.join(results_path, file))

    X = load_deepnet_activations(dnn_path, center=True, zscore=False, relu=True)
    Y = load_sparse_codes(embedding_path)

    print(f"\nShape of embedding matrix: {Y.shape}")
    print(f"Shape of DNN feature matrix: {X.shape}\n")

    assert (
        X.shape[0] == Y.shape[0]
    ), "\nNumber of objects in embedding and DNN feature matrix must be the same.\n"

    r2_scores = []
    for dim, y in enumerate(Y.T):
        cv_outer = KFold(n_splits=k_folds, shuffle=True, random_state=0)
        outer_alphas, outer_r2_scores, outer_models = [], [], []

        # This is the outer loop, where we split the data into k_folds
        for train_idx, test_idx in cv_outer.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            cv_inner = KFold(n_splits=k_folds, shuffle=True, random_state=1)

            model = Ridge(random_state=1)
            # Define search space, i.e. the hyperparameters to evaluate. Requires to have
            # intuition about appropriate search space beforehand.
            space = dict()
            space["alpha"] = np.linspace(50, 1200, 200)

            # Evluate the model on the inner loop
            search = GridSearchCV(
                model, space, scoring="r2", cv=cv_inner, refit=True, n_jobs=num_workers
            )
            result = search.fit(X_train, y_train)

            # Extract best params and estimators
            best_model = result.best_estimator_
            best_alpha = best_model.alpha

            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            outer_alphas.append(best_alpha)
            outer_models.append(best_model)

            outer_r2_scores.append(r2)

        r2_scores.append(np.mean(outer_r2_scores))
        print(f"Best alpha for dimension {dim}: {np.mean(outer_alphas)}")
        print(f"R2 score for dimension {dim}: {np.mean(outer_r2_scores)}\n")
        best_model = outer_models[np.argmax(outer_r2_scores)]
        joblib.dump(
            best_model, os.path.join(results_path, f"predictor_{dim:02d}.joblib")
        )

    with open(os.path.join(results_path, "r2_scores.npy"), "wb") as f:
        np.save(f, r2_scores)

    plot_predictions(r2_scores, results_path)


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    run_ridge_regression(
        dnn_path=args.dnn_path, embedding_path=args.embedding_path, k_folds=args.k_folds
    )
