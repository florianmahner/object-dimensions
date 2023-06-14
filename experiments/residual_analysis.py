import numpy as np
from object_dimensions import ExperimentParser
from object_dimensions.utils.utils import (
    load_sparse_codes,
    load_image_data,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


parser = ExperimentParser()
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


def main(args):
    print("Loading features and human embedding...")

    features = np.load(args.feature_path)
    features = np.maximum(features, 0)
    human_embedding = load_sparse_codes(args.human_path, relu=True)
    _, indices = load_image_data(args.img_root, filter_behavior=True)
    features = features[indices]

    # Fit a linear model to predict the human embedding from the DNN features.
    # Then, compute the residual between the predicted human embedding and the
    # actual human embedding.
    print("Fitting linear model...")

    for dependent_variable in ["human", "dnn"]:
        if dependent_variable == "human":
            X = features
            Y = human_embedding
        else:
            X = human_embedding
            Y = features

        k = 10
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        residuals_all = np.empty_like(Y)
        r2_scores = np.zeros(k)

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            model = LinearRegression()
            model.fit(X_train, Y_train)

            # Compute residuals for the current test fold
            Y_test_pred = model.predict(X_test)
            r2_scores[i] = model.score(X_test, Y_test)
            residuals_test = Y_test - Y_test_pred

            # Store residuals
            residuals_all[test_index] = residuals_test

        print("R2 scores across folds: ", r2_scores.mean(), r2_scores.std())
        residuals_all = np.abs(residuals_all)
        print(residuals_all.min())
        np.save(
            f"./data/misc/residuals_dependent_{dependent_variable}.npy", residuals_all
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
