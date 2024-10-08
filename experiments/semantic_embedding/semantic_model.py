# %%
import pandas as pd
import numpy as np
import glob
from scipy.io import loadmat
from objdim.utils import load_sparse_codes, load_image_data
import matplotlib.pyplot as plt

import json
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import joblib
import os
import seaborn as sns


MODEL_TO_RATINGS = {
    "vgg16_bn": "VGG-16",
    "human_behavior": "Human",
    "resnet50": "Resnet50",
    "OpenCLIP": "CLIP",
    "barlowtwins-rn50": "Barlow-Twins",
    "densenet": "Densenet",
}


def filter_nan_rows(X, sensevec):
    nan_rows = np.isnan(sensevec).any(axis=1)
    return X[~nan_rows], sensevec[~nan_rows]


def make_df_similarities(ratings):
    n_dimensions = len(ratings)
    # Create a DataFrame to store the results
    df_similarities = pd.DataFrame(
        {
            "Dimension": range(n_dimensions),
            "Type": [""] * n_dimensions,
        }
    )

    # Create a mapping of ratings to dimension types
    dimension_types = {
        "semantic": "Semantic",
        "visual": "Visual",
        "mix visual-semantic": "Mixed Visual-Semantic",
        "unclear": "Unclear",
    }

    # Assign types to dimensions using vectorized operations
    df_similarities["Type"] = pd.Series(ratings).map(dimension_types)

    return df_similarities


def load_and_preprocess_data(base_path, model_name, module_name):
    dnn = load_sparse_codes(base_path / f"data/embeddings/{model_name}/{module_name}/")
    humans = load_sparse_codes(base_path / "data/embeddings/human_behavior")
    images, indices = load_image_data(
        base_path / "data/images/things", filter_behavior=True
    )
    dnn = dnn[indices, :]

    df = pd.read_csv(base_path / "data/misc/dimension_ratings_processed.csv")

    index_name = MODEL_TO_RATINGS[model_name]
    dnn_labels = df[df["Model"] == index_name]
    dnn_labels.reset_index(drop=True, inplace=True)
    dnn_visual_or_semantic = dnn_labels["Quality"].to_list()

    human_labels = df[df["Model"] == "Human"]
    human_labels.reset_index(drop=True, inplace=True)
    human_visual_or_semantic = human_labels["Quality"].to_list()

    X = loadmat(base_path / "data/misc/sensevec.mat")
    X = X["sensevec"]

    dnn, sensevec = filter_nan_rows(dnn, X)
    humans, sensevec = filter_nan_rows(humans, X)

    # center predictor
    sensevec = sensevec - np.mean(sensevec, axis=0)

    return dnn, humans, sensevec, dnn_visual_or_semantic, human_visual_or_semantic


def predict_embedding_from_semantic(
    semantic_embedding,
    target_embedding,
    k_folds=5,
    num_workers=-1,
    results_path="./results",
):
    X = semantic_embedding
    Y = target_embedding

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    r2_scores = []
    predictions = []

    for dim, y in enumerate(Y.T):
        cv_outer = KFold(n_splits=k_folds, shuffle=True, random_state=0)
        outer_alphas, outer_r2_scores, outer_models = [], [], []

        for train_idx, test_idx in cv_outer.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            cv_inner = KFold(n_splits=k_folds, shuffle=True, random_state=1)

            model = Ridge(random_state=1)
            space = dict()
            space["alpha"] = np.logspace(-3, 5, 500)

            search = GridSearchCV(
                model, space, scoring="r2", cv=cv_inner, refit=True, n_jobs=num_workers
            )
            result = search.fit(X_train, y_train)

            best_model = result.best_estimator_
            best_alpha = best_model.alpha

            y_pred = best_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            outer_alphas.append(best_alpha)
            outer_models.append(best_model)
            outer_r2_scores.append(r2)

        r2_scores.append(np.mean(outer_r2_scores))

        print(f"Dimension {dim}:")
        print(f"  Best alpha: {np.mean(outer_alphas):.4f}")
        print(f"  R2 score: {r2_scores[-1]:.4f}")

        best_model = outer_models[np.argmax(outer_r2_scores)]
        joblib.dump(
            best_model,
            os.path.join(
                results_path, f"semantic_predictions_predictor_{dim:02d}.joblib"
            ),
        )

        # Generate predictions for the entire dataset
        dim_predictions = best_model.predict(X)
        predictions.append(dim_predictions)

    return r2_scores, np.array(predictions).T


def plot_results(df_similarities, results_path):
    plt.figure(figsize=(15, 6))
    sns.barplot(x="Dimension", y="R2_Score", hue="Type", data=df_similarities)
    plt.title("Dimension Prediction R2 Scores")
    plt.savefig(os.path.join(results_path, "dimension_prediction_r2_scores.png"))
    # plt.close()

    plt.figure(figsize=(15, 6))
    sns.barplot(x="Dimension", y="Sparsity_R2_Score", hue="Type", data=df_similarities)
    plt.title("Sparsity Prediction R2 Scores")
    plt.savefig(os.path.join(results_path, "sparsity_prediction_r2_scores.png"))
    # plt.close()

    plt.figure(figsize=(15, 6))
    sns.barplot(x="Dimension", y="Residual_R2_Score", hue="Type", data=df_similarities)
    plt.title("Residual Prediction R2 Scores")
    plt.savefig(os.path.join(results_path, "residual_prediction_r2_scores.png"))
    # plt.close()


def get_model_names():
    files = glob.glob("/LOCAL/fmahner/object-dimensions/data/embeddings/*/*/")

    model_names, module_names = [], []
    for file in files:
        model_name = file.split("/")[-3].replace("-", "_")
        module_name = file.split("/")[-2].replace("-", "_")
        if model_name == "human_behavior":
            continue

        if module_name in ["features.12", "features.5", "features.42", "features.22"]:
            continue
        model_names.append(model_name)
        module_names.append(module_name)

    return model_names, module_names


# %%
base_path = Path("/LOCAL/fmahner/object-dimensions")

r2_scores = dict()
dnn_embeddings = dict()

model_names, module_names = get_model_names()

# %%

dnn_embedding, human_embedding, sensevec, dnn_ratings, human_ratings = (
    load_and_preprocess_data(base_path, model_names[0], module_names[0])
)

# %%

# Calculate dim prediction for humans
human_results_path = base_path / "results/semantic_to_humans"
human_r2, human_predictions = predict_embedding_from_semantic(
    sensevec, human_embedding, results_path=human_results_path
)

for model_name, module_name in zip(model_names, module_names):
    dnn_embedding, human_embedding, sensevec, dnn_ratings, human_ratings = (
        load_and_preprocess_data(base_path, model_name, module_name)
    )

    dnn_results_path = base_path / f"results/semantic_to_{model_name}"

    dnn_r2, dnn_predictions = predict_embedding_from_semantic(
        sensevec, dnn_embedding, results_path=dnn_results_path
    )
    r2_scores[f"{model_name}_r2"] = dnn_r2
    dnn_embeddings[f"{model_name}_embedding"] = dnn_embedding


# %%

# rename the existing r2_scores and dnn_embeddings

# This code doesn't work in place because it modifies the dictionary while iterating over it.
# Here's a corrected version that works in place:


r2_scores = {key + "_r2": value for key, value in list(r2_scores.items())}
dnn_embeddings = {
    key + "_embedding": value for key, value in list(dnn_embeddings.items())
}

# %%
import scipy.io

# make a mat file of sensevec embedding, human_embedding, dnn_embedding
mat_path = base_path / "sensevec_human_dnn_embeddings.mat"
# scipy.io.savemat(
#     mat_path,
#     {
#         "sensevec": sensevec,
#         "human_embedding": human_embedding,
#         "human_r2": human_r2,
#         **r2_scores,
#         **dnn_embeddings,
#     },
# )

# %%

# get the dnn ratings for all models
dnn_ratings_all = dict()
for model_name, module_name in zip(model_names, module_names):
    _, _, _, dnn_ratings, _ = load_and_preprocess_data(
        base_path, model_name, module_name
    )
    dnn_ratings_all[f"{model_name}"] = make_df_similarities(dnn_ratings)

# %%
#
results_path = base_path / "results/semantic_to_all_models"
os.makedirs(results_path, exist_ok=True)
# load scipy mat
mat = scipy.io.loadmat(mat_path)
# Define a consistent color palette for the hue types
color_palette = {
    "Semantic": "#1f77b4",
    "Visual": "#ff7f0e",
    "Mixed Visual-Semantic": "#2ca02c",
    "Unclear": "#d62728",
}

# Define the order for the legend
hue_order = ["Semantic", "Visual", "Mixed Visual-Semantic", "Unclear"]

# Create one large figure
fig, axes = plt.subplots(
    len(model_names), 1, figsize=(15, 8 * len(model_names)), squeeze=False
)
fig.suptitle("R2 Scores for All Models", fontsize=16)

for idx, model_name in enumerate(model_names):
    r2 = mat[f"{model_name}_r2"]

    df = dnn_ratings_all[f"{model_name}"]
    df["R2_Score"] = r2.ravel()

    ax = axes[idx, 0]
    sns.barplot(
        x="Dimension",
        y="R2_Score",
        data=df,
        hue="Type",
        palette=color_palette,
        hue_order=hue_order,
        ax=ax,
    )
    ax.set_title(f"{model_name} R2 Scores")
    ax.legend(title="Type", loc="upper right")

    # Only show x-axis labels for the bottom subplot
    if idx < len(model_names) - 1:
        ax.set_xticklabels([])
        ax.set_xlabel("")

plt.tight_layout()
plt.savefig(os.path.join(results_path, "all_models_r2_scores.png"))
plt.close()


# %%


# # plot a linear regression line for human_sum vs human_r2


def center(X):
    return X - np.mean(X)


sqrt_tsfm = lambda x: np.sqrt(x)


def fit_linreg(x, y):
    fit = np.polyfit(x, y, 1)
    coef, intercept = fit
    return coef, intercept


def plot_linreg(x, y, coef, intercept):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y)
    plt.plot(x, coef * x + intercept, color="red")
    plt.xlabel("Sum of Embedding")
    plt.ylabel("R2 Score")


human_centered_r2 = center(human_r2)
dnn_centered_r2 = center(dnn_r2)
sum_human_embedding = sqrt_tsfm(np.sum(human_embedding, axis=0))
sum_dnn_embedding = sqrt_tsfm(np.sum(dnn_embedding, axis=0))

coef_human, intercept_human = fit_linreg(sum_human_embedding, human_centered_r2)
coef_dnn, intercept_dnn = fit_linreg(sum_dnn_embedding, dnn_centered_r2)

plot_linreg(sum_human_embedding, human_centered_r2, coef_human, intercept_human)
plot_linreg(sum_dnn_embedding, dnn_centered_r2, coef_dnn, intercept_dnn)


# %%

predictions_human = (
    coef_human * sqrt_tsfm(np.sum(human_embedding, axis=0)) + intercept_human
)
predictions_dnn = coef_dnn * sqrt_tsfm(np.sum(dnn_embedding, axis=0)) + intercept_dnn

residuals_human = human_centered_r2 - predictions_human
residuals_dnn = dnn_centered_r2 - predictions_dnn


# %%

# Von martin

mean_dnn_r2 = np.mean(dnn_r2)
mean_human_r2 = np.mean(human_r2)

dnn_r2_centered = dnn_r2 - mean_dnn_r2
human_r2_centered = human_r2 - mean_human_r2

sparseness_dnn = np.sum(dnn_embedding, axis=0)
sparseness_human = np.sum(human_embedding, axis=0)

sparseness_dnn_centered = sparseness_dnn - np.mean(sparseness_dnn)
sparseness_human_centered = sparseness_human - np.mean(sparseness_human)

# Compute regression coefficients
b_human_r2 = np.sum(sparseness_human_centered * human_r2_centered) / np.sum(
    sparseness_human_centered**2
)
b_dnn_r2 = np.sum(sparseness_dnn_centered * dnn_r2_centered) / np.sum(
    sparseness_dnn_centered**2
)

# Compute residuals
residuals_human_r2 = (
    human_r2_centered - b_human_r2 * sparseness_human_centered + mean_human_r2
)
residuals_dnn_r2 = dnn_r2_centered - b_dnn_r2 * sparseness_dnn_centered + mean_dnn_r2

# %%

# make a bar plot of residuals_human_r2 and in the df_similarity_humans
df_similarity_humans = make_df_similarities(human_ratings)
df_similarity_humans["Residual_R2_Score"] = residuals_human_r2

plt.figure(figsize=(10, 6))
sns.barplot(
    x="Dimension",
    y="Residual_R2_Score",
    data=df_similarity_humans,
    hue="Type",
)
plt.title("Human Residual R2 Scores")
plt.savefig(os.path.join(human_results_path, "human_residual_r2_scores.png"))


# and for dnn
plt.figure(figsize=(10, 6))
df_similarity_dnn = make_df_similarities(dnn_ratings)
df_similarity_dnn["Residual_R2_Score"] = residuals_dnn_r2

sns.barplot(
    x="Dimension",
    y="Residual_R2_Score",
    data=df_similarity_dnn,
    hue="Type",
)
plt.title("DNN Residual R2 Scores")
plt.savefig(os.path.join(dnn_results_path, "dnn_residual_r2_scores.png"))

# %%


# %%

df_similarity_humans = make_df_similarities(human_ratings)

# df_similarity_humans["R2_Score"] = predictions_human
df_similarity_humans["Residual_R2_Score"] = residuals_human
df_similarity_humans["R2_Score"] = human_r2

sns.barplot(
    x="Dimension",
    y="Residual_R2_Score",
    data=df_similarity_humans,
    hue="Type",
)

plot_results(df_similarity_humans, human_results_path)

# %%


# %%

load_mat = scipy.io.loadmat(mat_path)


# %%

# change the key of - into _ and save the mat file again
load_mat = {key.replace("-", "_"): value for key, value in load_mat.items()}
scipy.io.savemat(mat_path, load_mat)

# %%

# Create DataFrames for plotting
df_similarity_dnn = make_df_similarities(human_ratings)
df_similarity_dnn["R2_Score"] = dnn_r2
df_similarity_dnn["Sparsity_R2_Score"] = dnn_sparsity_r2
df_similarity_dnn["Residual_R2_Score"] = dnn_residual_r2


df_similarity_humans = make_df_similarities(human_visual_or_semantic)
df_similarity_humans["R2_Score"] = human_r2
df_similarity_humans["Sparsity_R2_Score"] = human_sparsity_r2
df_similarity_humans["Residual_R2_Score"] = human_residual_r2


# %%
# Plot results
plot_results(df_similarity_dnn, dnn_results_path)


# %%

plot_results(df_similarity_humans, human_results_path)
