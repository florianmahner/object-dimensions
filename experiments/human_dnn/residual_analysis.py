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
    parser.add_argument(
        "--shared_type",
        type=str,
        default="residual",
        choices=["residuals", "intersection"],
        help="Type of analysis to run. Residual is y-y_hat, shared is y_hat (intersection).",
    )
    return parser.parse_args()


global NUM_WORKERS
NUM_WORKERS = multiprocessing.cpu_count() - 1


@dataclass
class _Regression(object):
    weights: np.ndarray
    bias: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weights + self.bias

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(x)
        return r2_score(y, y_pred)


def search_optimal_regularization(
    X: np.ndarray,
    Y: np.ndarray,
    k_folds: int,
    out_path: str = None,
) -> Tuple[List, List, List]:
    r2_scores, models, regularizers = [], [], []
    n_features, n_targets = X.shape[1], Y.shape[1]
    weights = np.zeros((n_features, n_targets))
    bias = np.zeros(n_targets)

    pbar = tqdm.tqdm(range(n_targets))

    for i, y in enumerate(Y.T):
        pbar.update(1)
        cv_outer = KFold(n_splits=k_folds, shuffle=True, random_state=0)
        outer_regularizer, outer_r2_scores, outer_models = [], [], []

        # This is the outer loop, where we split the data into k_folds
        for train_idx, test_idx in cv_outer.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            cv_inner = KFold(n_splits=k_folds, shuffle=True, random_state=1)

            model = Ridge(random_state=1)
            # Define search space, i.e. the hyperparameters to evaluate. Requires to have
            # intuition about appropriate search space beforehand.
            space = dict()
            space["alpha"] = np.linspace(100, 4000, 200)

            # Evluate the model on the inner loop
            search = GridSearchCV(
                model, space, scoring="r2", cv=cv_inner, refit=True, n_jobs=NUM_WORKERS
            )
            result = search.fit(X_train, y_train)

            # Extract best params and estimators
            best_model = result.best_estimator_
            best_alpha = best_model.alpha

            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            outer_regularizer.append(best_alpha)
            outer_models.append(best_model)
            outer_r2_scores.append(r2)

        max_r2 = np.argmax(outer_r2_scores)
        alpha = outer_regularizer[max_r2]
        print(f"Best alpha for dimension {i}: {alpha}")
        r2_scores.append(np.mean(outer_r2_scores))

        best_model = outer_models[max_r2]

        weights[:, i] += best_model.coef_
        bias[i] += best_model.intercept_

    model = _Regression(
        weights,
        bias,
    )
    return model, r2_scores


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
    k: int = 10,
    shared_type: str = "residuals",
    out_path: str = "./data/misc/rsa",
    run_search: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    fname = os.path.join(out_path, "cv_residual_model.pkl")
    if run_search:
        model, r2_scores = search_optimal_regularization(
            X, Y, k_folds=k, out_path=out_path
        )
        with open(fname, "wb") as f:
            pickle.dump(model, f)
        plot_r2(r2_scores, out_path)
    else:
        assert os.path.exists(fname), "Model not found. Please run search first."
        with open(fname, "rb") as f:
            model = pickle.load(f)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    predictions_all = np.empty_like(Y)
    r2_scores = np.zeros(k)

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        Y_test_pred = model.predict(X_test)
        r2_scores[i] = model.score(X_test, Y_test)
        if shared_type == "intersection":
            pred_test = Y_test_pred
        else:
            # Compute residuals for the current test fold
            residuals = Y_test - Y_test_pred
            pred_test = residuals

        # Store residuals
        predictions_all[test_index] = pred_test
    predictions_all -= np.min(predictions_all)

    return predictions_all, r2_scores


def main(
    human_path: str,
    feature_path: str,
    img_root: str,
    k: int,
    shared_type: str,
    run_search: bool = False,
) -> None:
    print("Loading features and human embedding...")

    features = load_deepnet_activations(feature_path, relu=True, center=True)
    human_embedding = load_sparse_codes(human_path, relu=True)
    indices = load_image_data(img_root, filter_behavior=True)[1]
    features = features[indices]
    print("Fitting linear model...")
    out_path = "./data/misc/rsa"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for dependent_variable in ["human", "dnn"]:
        if dependent_variable == "human":
            X = features
            Y = human_embedding
        else:
            X = human_embedding
            Y = features

        predictions_all, r2_scores = cross_validated_prediction(
            X, Y, k=k, shared_type=shared_type, run_search=run_search, out_path=out_path
        )

        print(
            "R2 scores across folds: ",
            r2_scores.mean(),
            r2_scores.std(),
            "dependent_variable: ",
            dependent_variable,
        )

        fname = os.path.join(
            out_path, f"{shared_type}_dependent_{dependent_variable}.npy"
        )
        np.save(
            fname,
            predictions_all,
        )


if __name__ == "__main__":
    args = parse_args()

    main(
        args.human_path,
        args.feature_path,
        args.img_root,
        args.k_folds,
        args.shared_type,
        args.run_search,
    )
