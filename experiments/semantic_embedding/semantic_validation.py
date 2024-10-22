# %%
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any


def load_data() -> Tuple[Dict[str, Any], pd.DataFrame]:
    mat_data = loadmat(
        "/LOCAL/fmahner/object-dimensions/data/misc/sensevec_human_dnn_embeddings.mat"
    )
    dimension_ratings = pd.read_csv(
        "/LOCAL/fmahner/object-dimensions/data/misc/dimension_ratings_processed.csv"
    )
    return mat_data, dimension_ratings


def create_models_dict(mat_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    model_configs = [
        ("Human", "human_r2"),
        ("VGG-16", "vgg16_bn_r2"),
        ("Resnet50", "resnet50_r2"),
        ("Densenet", "densenet_r2"),
        ("CLIP", "OpenCLIP_r2"),
        ("Barlow-Twins", "barlowtwins_rn50_r2"),
    ]

    return {
        name: {"r2": mat_data[key].flatten(), "label": name}
        for name, key in model_configs
    }


def remove_sparsity(r2: np.ndarray) -> np.ndarray:
    """Remove sparsity from R2 values by fitting a linear model and removing the trend.
    This effectively removes the effect of the linear relationship between the model and human ratings,
    ie that early dimensions are more predictive of the human ratings and later dimensions are less
    predictive. After this subtraction, the values above the linear line are the important ones.
    """
    n = len(r2)
    mean_r2 = np.mean(r2)
    r2 -= mean_r2
    x = np.linspace(-1, 1, n)
    slope = np.linalg.lstsq(x.reshape(-1, 1), r2, rcond=None)[0][0]
    residuals = r2 - slope * x
    return residuals + mean_r2


def process_model(
    model_data: Dict[str, Any],
    dimension_ratings: pd.DataFrame,
    model_name: str,
    normalize: bool = True,
) -> pd.DataFrame:
    model_ratings = dimension_ratings[dimension_ratings["Model"] == model_data["label"]]

    valid_labels = ["semantic", "visual"]

    # Filter for valid labels and set 'Dimension' as index
    valid_ratings = model_ratings[
        model_ratings["Quality"].isin(valid_labels)
    ].set_index("Dimension")

    # Get R2 values
    r2 = model_data["r2"]

    if len(r2) != len(model_ratings):
        raise ValueError("Length of r2 does not match the number of ratings entries")

    if normalize:
        r2 = remove_sparsity(r2)

    # Filter R2 values for valid labels
    r2_valid = r2[valid_ratings.index]

    thresholds = np.linspace(0, 1, 100)
    results = []

    for threshold in thresholds:
        predicted_visual = r2_valid < threshold
        true_visual = valid_ratings["Quality"].isin(["visual"]).values

        tn, fp, fn, tp = confusion_matrix(true_visual, predicted_visual).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        balanced_accuracy = (sensitivity + specificity) / 2

        results.append(
            {
                "model": model_name,
                "threshold": threshold * 100,
                "accuracy": accuracy * 100,
                "balanced_accuracy": balanced_accuracy * 100,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "fpr": (1 - specificity) * 100,
                "tpr": sensitivity * 100,
            }
        )

    return pd.DataFrame(results)


def plot_results(results_df: pd.DataFrame) -> Tuple[plt.Figure, plt.Figure]:
    palette = sns.color_palette("deep", n_colors=len(results_df["model"].unique()))
    # Accuracy plot
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.lineplot(
        data=results_df,
        x="threshold",
        y="balanced_accuracy",
        hue="model",
        ax=ax1,
        palette=palette,
    )

    # Calculate and plot the mean accuracy across all models
    mean_accuracy = (
        results_df.groupby("threshold")["balanced_accuracy"].mean().reset_index()
    )
    sns.lineplot(
        data=mean_accuracy,
        x="threshold",
        y="balanced_accuracy",
        ax=ax1,
        color="black",
        linewidth=2,
        label="Mean",
    )

    ax1.set(
        xlabel="Threshold [$R^2$]",
        ylabel="Balanced Accuracy [%]",
    )
    ax1.legend(title="Model")

    # ROC plot
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.lineplot(
        data=results_df,
        x="fpr",
        y="tpr",
        hue="model",
        ax=ax2,
        errorbar=None,
        palette=palette,
    )

    # for each model i have a tpr and fpr at an index. make the mean of these
    # Calculate mean ROC curve
    model_groups = results_df.groupby("model")
    mean_tpr = model_groups["tpr"].apply(lambda x: x.values).mean()
    mean_fpr = model_groups["fpr"].apply(lambda x: x.values).mean()

    # Interpolate to ensure all FPR values are present
    interp_fpr = np.linspace(0, 1, 100)
    interp_tpr = np.interp(interp_fpr, mean_fpr, mean_tpr)

    mean_roc = pd.DataFrame({"fpr": interp_fpr, "tpr": interp_tpr})

    sns.lineplot(
        x=mean_fpr,
        y=mean_tpr,
        ax=ax2,
        color="black",
        linewidth=2,
        label="Mean",
    )

    ax2.plot([0, 100], [0, 100], color="black", linestyle="--")
    ax2.set(xlabel="False Positive Rate [%]", ylabel="True Positive Rate [%]")
    ax2.legend(title="Model")
    ax2.set_xlim(0, 101)
    ax2.set_ylim(0, 101)

    plt.tight_layout()
    # despine all plots using seaborn
    sns.despine(fig1)
    sns.despine(fig2)
    return fig1, fig2


def plot_r2_scores(model_r2_data: Dict[str, np.ndarray], threshold: float = 0.3):
    r2_df = pd.DataFrame(
        [
            (model_name, model_data["r2"])
            for model_name, model_data in model_r2_data.items()
        ],
        columns=["model", "r2"],
    )
    r2_df = r2_df.explode("r2").reset_index(drop=True)
    r2_df = r2_df[r2_df["r2"] > threshold]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set(xlabel="Model", ylabel="Mean $R^2$ score")
    sns.barplot(x="model", y="r2", data=r2_df, ax=ax)
    sns.despine(fig)
    return fig


# def main():

# %%
base_path = Path("/LOCAL/fmahner/object-dimensions")
mat_data, dimension_ratings = load_data()
models = create_models_dict(mat_data)
results_df = pd.concat(
    [
        process_model(model_data, dimension_ratings, model_name)
        for model_name, model_data in copy.deepcopy(models).items()
    ],
    ignore_index=True,
)

# %%
fig = plot_r2_scores(copy.deepcopy(models), threshold=-1)
fig.savefig(
    base_path / "results" / "semantic_validation" / "r2_scores.pdf",
    bbox_inches="tight",
    dpi=300,
)
# %%
fig = plot_results(results_df)


fig[0].savefig(
    base_path / "results" / "semantic_validation" / "accuracy_semantic_validation.pdf",
    bbox_inches="tight",
    dpi=300,
)
fig[1].savefig(
    base_path / "results" / "semantic_validation" / "roc_semantic_validation.pdf",
    bbox_inches="tight",
    dpi=300,
)


# if __name__ == "__main__":
# main()
