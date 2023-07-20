import tqdm
import os
import pickle
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from tomlparse import argparse
from dataclasses import dataclass
from object_dimensions.utils.utils import (
    load_sparse_codes,
    load_image_data,
    load_deepnet_activations,
)
from typing import Tuple, List, Union
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score

global NUM_WORKERS
NUM_WORKERS = multiprocessing.cpu_count() - 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_path",
        type=str,
        default="./features",
        help="Path to DNN features directory",
    )
    parser.add_argument("--human_path", type=str, help="Path to human embedding")
    parser.add_argument(
        "--img_root",
        type=str,
        default="./data/images",
        help="Path to the all images used for the embedding.",
    )
    parser.add_argument("--k_folds", type=int, default=4, help="Number of folds.")
    parser.add_argument(
        "--run_search",
        action="store_true",
        help="Run crossvalidated search of regularization parameter.",
    )
    return parser.parse_args()


def plot_r2(r2_scores: Union[List, np.ndarray], out_path: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(r2_scores)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("R2 score")
    ax.set_title("R2 scores for residuals")
    fig.savefig(os.path.join(out_path, "r2_scores.png"), dpi=300)
    plt.close(fig)


def cross_validated_prediction(
    X: np.ndarray,
    Y: np.ndarray,
    k_folds: int,
    out_path: str = None,
) -> Tuple[List, List, List]:
    r2_scores = []

    n_objects, n_features = X.shape
    n_targets = Y.shape[1]
    pbar = tqdm.tqdm(range(n_targets))
    residuals = np.zeros((n_objects, n_targets))
    intersection = np.zeros((n_objects, n_targets))
    r2_scores = np.zeros(n_targets)

    for i, y in enumerate(Y.T):
        pbar.update(1)
        cv_outer = KFold(n_splits=k_folds, shuffle=True, random_state=0)
        outer_r2_scores = []
        # This is the outer loop, where we split the data into k_folds
        for train_idx, test_idx in cv_outer.split(X):
            print("Outer loop")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Define search space, i.e. the hyperparameters to evaluate. Requires to have
            # intuition about appropriate search space beforehand.
            search_space = dict()
            search_space["alpha"] = np.linspace(100, 4000, 100)

            cv_inner = KFold(n_splits=k_folds, shuffle=True, random_state=1)
            model = Ridge(random_state=1)
            search = GridSearchCV(
                model, search_space, scoring="r2", cv=cv_inner, refit=True
            )

            results = search.fit(X_train, y_train)
            best_model = results.best_estimator_
            y_hat = best_model.predict(X_test)

            residuals[test_idx, i] = y_test - y_hat
            intersection[test_idx, i] = y_hat

            r2_score = best_model.score(X_test, y_test)
            outer_r2_scores.append(r2_score)

        r2_scores[i] = np.mean(outer_r2_scores)

    # Make our predicitions positive
    residuals = residuals + np.abs(np.min(residuals))
    intersection = intersection + np.abs(np.min(intersection))
    return residuals, intersection, r2_scores


def load_data(
    human_path: str, feature_path: str, img_root: str
) -> Tuple[np.ndarray, np.ndarray]:
    features = load_deepnet_activations(feature_path, relu=True, center=True)
    human_embedding = load_sparse_codes(human_path, relu=True)
    indices = load_image_data(img_root, filter_behavior=True)[1]
    features = features[indices]
    return features, human_embedding


def run(
    human_path: str,
    feature_path: str,
    img_root: str,
    k: int,
    run_search: bool = False,
) -> None:
    out_path = "./data/misc/rsa"
    features, human_embedding = load_data(human_path, feature_path, img_root)
    residuals, intersection, r2_scores = cross_validated_prediction(
        features,
        human_embedding,
        k,
        out_path,
    )
    plot_r2(r2_scores, out_path)
    print("Average r2 score across all dimensions: ", np.mean(r2_scores))

    for predictions, name in zip(
        [residuals, intersection], ["residuals", "intersection"]
    ):
        fname = os.path.join(out_path, f"{name}_dependent_human.npy")
        np.save(fname, predictions)


if __name__ == "__main__":
    args = parse_args()
    run(
        args.human_path,
        args.feature_path,
        args.img_root,
        args.k_folds,
        args.run_search,
    )
