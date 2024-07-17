""" This file analyzes the dimension ratings provided by human annotators to label the dimensions of the DNN and human embeddings"""

import json
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from scipy.io import loadmat
from objdim.utils import load_sparse_codes
from tomlparse import argparse

# Identifiers for the models in the ratings
MODEL_TO_DIMENSIONS = {
    1: "Human",
    2: "Resnet50",
    3: "CLIP",
    4: "Densenet",
    5: "Barlow-Twins",
    6: "VGG-16",
}

ANSWER_TO_QUALITY = {
    1: "visual",
    2: "semantic",
    3: "mix visual-semantic",
    4: "unclear",
}

ANSWER_TO_CONCEPT = {
    1: "single concept",
    2: "multiple concepts",
    3: "uninterpretable",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dimension_mapping_path",
        type=str,
        default="./data/misc/dimension_mapping.json",
        help="Path to the json file containing the mapping of the anonymized dimensions to the original dimensions",
    )
    parser.add_argument(
        "--dimension_rating_path",
        type=str,
        default="./data/misc/dimension_ratings.mat",
        help="Path to the mat file containing the human ratings",
    )
    return parser.parse_args()


def plot_quality(df):
    df = df[df["Quality"] != "unclear"]

    quality_counts = (
        df.groupby(["Model", "Quality"], observed=False).size().unstack(fill_value=0)
    )
    quality_percent = quality_counts.div(
        quality_counts.sum(axis=1), axis=0
    ).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))

    # Initialize the left position for stacking
    left = pd.Series([0] * len(quality_percent), index=quality_percent.index)

    # List of qualities for consistent ordering
    qualities_list = ["semantic", "mix visual-semantic", "visual"]

    # Default seaborn color palette
    palette = sns.color_palette("deep")

    # Map specific colors to each quality using default palette
    quality_to_color = {
        "visual": palette[2],  # dark blue
        "mix visual-semantic": palette[1],  # light blue
        "semantic": palette[0],  # orange
    }

    # Plot each quality as a separate bar
    for quality in qualities_list:
        sns.barplot(
            y="Model",
            x=quality,
            data=quality_percent,
            ax=ax,
            label=quality,
            color=quality_to_color[quality],
            left=left,
        )
        left += quality_percent[quality]

    # put the legend on top of the plot with, eg similar to a title and with pad inches
    ax.legend(
        ncol=len(qualities_list),
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 1.15),
    )

    ax.set(xlabel="Percentage of Ratings", ylabel="Model")
    sns.despine(left=True, bottom=True)
    fig.savefig(
        "./results/plots/dimension_quality_ratings.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def plot_concept(df):
    # Counting occurrences of each concept per model and convert to percentage
    concept_counts = (
        df.groupby(["Model", "Concept"], observed=False).size().unstack(fill_value=0)
    )
    concept_percent = concept_counts.div(
        concept_counts.sum(axis=1), axis=0
    ).reset_index()

    # Plotting the stacked bar plot for concept ratings using seaborn and matplotlib
    fig, ax = plt.subplots(figsize=(6, 4))

    # Initialize the left position for stacking
    left = pd.Series([0] * len(concept_percent), index=concept_percent.index)

    # List of concepts for consistent ordering
    concepts_list = ["single concept", "multiple concepts", "uninterpretable"]

    # Default seaborn color palette for concepts
    palette_concepts = sns.color_palette("deep")

    # Map specific colors to each concept using default palette
    concept_to_color = {
        "single concept": palette_concepts[0],  # blue
        "multiple concepts": palette_concepts[1],  # green
        "uninterpretable": palette_concepts[2],  # red
    }

    # Plot each concept as a separate bar
    for concept in concepts_list:
        sns.barplot(
            y="Model",
            x=concept,
            data=concept_percent,
            ax=ax,
            label=concept,
            color=concept_to_color[concept],
            left=left,
        )
        left += concept_percent[concept]

    ax.legend(
        ncol=len(concepts_list),
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 1.15),
    )
    ax.set(xlabel="Fraction of Ratings", ylabel="Model")
    sns.despine(left=True, bottom=True)
    fig.savefig(
        "./results/plots/dimension_concept_ratings.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def plot_concept_bar(df):
    # Counting occurrences of each concept per model and convert to percentage
    concept_counts = (
        df.groupby(["Model", "Concept"], observed=False).size().unstack(fill_value=0)
    )
    concept_percent = concept_counts.div(
        concept_counts.sum(axis=1), axis=0
    ).reset_index()

    # only plot the uninterpretable concept
    concept_percent = concept_percent[["Model", "uninterpretable"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x="Model", y="uninterpretable", data=concept_percent, ax=ax, color="grey"
    )
    # make patches of lines inside the bar
    for i, bar in enumerate(ax.patches):
        bar.set_hatch("//")
        bar.set_edgecolor("black")

    # ax.tick_params(axis="x", rotation=45)

    ax.set(xlabel="Model", ylabel="Fraction of Uninterpretable Dimensions")
    sns.despine(left=True, bottom=True)
    fig.savefig(
        "./results/plots/dimension_uninterpretable_ratings.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def load_dimension_mapping(path: str):
    """We have anonymized each dimension in the human experiments by shuffling the dimension order
    to ensure that the human ratings are not biased by the order of the dimensions. This function loads
    the correct mapping of the anonymized dimensions to the original dimensions."""
    dimension_mapping = json.load(open(path))

    new_names = {
        "barlow": "Barlow-Twins",
        "resnet": "Resnet50",
        "densenet": "Densenet",
        "vgg": "VGG-16",
        "clip": "CLIP",
        "human": "Human",
    }

    new_mapping = {}
    for key, val in dimension_mapping.items():
        new_key = new_names[key]
        new_mapping[new_key] = val

    dimension_mapping = new_mapping
    return dimension_mapping


def reorder_df(df, ordering):
    """Function to reorder DataFrame based on ordering dictionary"""
    # Ensure the dictionary keys and values are integers
    ordering = {int(k): int(v) for k, v in ordering.items()}

    # Create a list of row indices based on the values in the dictionary
    new_order = sorted(ordering, key=ordering.get)

    # Check if the DataFrame length matches the ordering length
    if len(new_order) > len(df):
        new_order = new_order[: len(df)]

    # Reorder the DataFrame using these indices
    return df.iloc[new_order].reset_index(drop=True)


def process_dimension_ratings(ratings, dimension_mapping):
    models = ratings[:, 0]
    qualities = ratings[:, 1]
    concepts = ratings[:, 4]

    df = pd.DataFrame({"Model": models, "Quality": qualities, "Concept": concepts})

    # we map the model the integer ratings to the correct names
    df["Model"] = df["Model"].map(MODEL_TO_DIMENSIONS)
    df["Quality"] = df["Quality"].map(ANSWER_TO_QUALITY)
    df["Concept"] = df["Concept"].map(ANSWER_TO_CONCEPT)

    # Split the DataFrame by model
    models = df["Model"].unique()
    reordered_dfs = []

    for model in models:
        model_df = df[df["Model"] == model]

        reordered_model_df = reorder_df(model_df, dimension_mapping[model])
        reordered_dfs.append(reordered_model_df)

    # Concatenate the reordered DataFrames back into a single DataFrame
    reordered_df = pd.concat(reordered_dfs).reset_index(drop=True)
    model_order = ["Human", "VGG-16", "Resnet50", "CLIP", "Densenet", "Barlow-Twins"]
    reordered_df["Model"] = pd.Categorical(
        reordered_df["Model"], categories=model_order, ordered=True
    )

    df = reordered_df
    df["Dimension"] = df.groupby("Model", observed=False).cumcount()
    return df


def print_percentage_concept(df):
    for model in df["Model"].unique():
        for concept in df["Concept"].unique():
            model_df = df[df["Model"] == model]
            concept_df = model_df[model_df["Concept"] == concept]
            percentage = len(concept_df) / len(model_df) * 100
            print(f"Percentage of {concept} ratings in {model}: {percentage:.2f}%")


def print_percentage_quality(df):
    for model in df["Model"].unique():
        for quality in df["Quality"].unique():
            model_df = df[df["Model"] == model]
            quality_df = model_df[model_df["Quality"] == quality]
            percentage = len(quality_df) / len(model_df) * 100
            print(f"Percentage of {quality} ratings in {model}: {percentage:.2f}%")


def plot_semantic_visual(df):

    human = df[df["Model"] == "Human"]
    vgg16 = df[df["Model"] == "VGG-16"]

    # Remove 'unclear' quality ratings
    human = human[human["Quality"] != "unclear"]
    vgg16 = vgg16[vgg16["Quality"] != "unclear"]

    # Count occurrences of each quality and convert to percentages
    percentage_human = human["Quality"].value_counts(normalize=True).reset_index()
    percentage_human.columns = ["Quality", "Percentage"]
    percentage_human["Model"] = "Human"

    percentage_vgg16 = vgg16["Quality"].value_counts(normalize=True).reset_index()
    percentage_vgg16.columns = ["Quality", "Percentage"]
    percentage_vgg16["Model"] = "VGG-16"

    # Combine data into one DataFrame
    percentage_df = pd.concat([percentage_human, percentage_vgg16])

    sns.set_context("paper")

    # Plotting the stacked bar plot using seaborn and matplotlib
    fig, ax = plt.subplots(figsize=(9, 3))

    # List of qualities for consistent ordering
    qualities_list = ["semantic", "mix visual-semantic", "visual"]

    # Default seaborn color palette
    palette = sns.color_palette("deep")

    # Map specific colors to each quality using default palette
    quality_to_color = {
        "visual": palette[2],  # dark blue
        "mix visual-semantic": palette[1],  # light blue
        "semantic": palette[0],  # orange
    }

    # Create a pivot table for easier plotting
    percentage_pivot = percentage_df.pivot(
        index="Model", columns="Quality", values="Percentage"
    ).fillna(0)

    # Initialize the left position for stacking
    left = pd.Series([0] * len(percentage_pivot), index=percentage_pivot.index)

    # Plot each quality as a separate bar
    for quality in qualities_list:
        sns.barplot(
            y=percentage_pivot.index,
            x=percentage_pivot[quality],
            ax=ax,
            label=quality,
            color=quality_to_color[quality],
            left=left,
        )
        left += percentage_pivot[quality]

    # Put the legend on top of the plot
    ax.legend(
        ncol=len(qualities_list),
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 1.15),
    )

    ax.set(xlabel="Percentage of Ratings", ylabel="Model")
    sns.despine(left=True, bottom=True)
    fig.savefig(
        "./results/plots/human_vgg16_quality_ratings.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def calculate_total_sum_of_embedding(W):
    sum_across_dimensions = W.sum(axis=0)
    total_sum = sum_across_dimensions.sum()
    variance_per_dimension = sum_across_dimensions / total_sum
    return variance_per_dimension


def weighted_uninterpretability(embeddings, df):
    """this function weights for a specific embedding if that dimension is labeled as uninterpretable by the sum
    of weights of that dimension. we can think of this analysis as the relevance of the uninterpretable dimensions
    ie the explained variance of the uninterpretable dimensions in the embedding."""

    variances = {}
    for name, embedding in embeddings.items():
        variance = calculate_total_sum_of_embedding(embedding)
        model = df[df["Model"] == name]
        uninterpretable = model[model["Concept"] == "uninterpretable"]
        uninterpretable_variance = variance[uninterpretable["Dimension"]].sum()
        variances[name] = uninterpretable_variance * 100

    # here is the plotting part now.
    variance = pd.DataFrame(
        variances.items(), columns=["Model", "Uninterpretable Variance"]
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x="Model", y="Uninterpretable Variance", data=variance, ax=ax, color="grey"
    )
    ax.set(
        xlabel="Model",
        ylabel="Variance in Embedding Explained\nby Uninterpretable Dimensions [%]",
    )
    sns.despine(left=True, bottom=True)
    # make hatch
    for i, bar in enumerate(ax.patches):
        bar.set_hatch("//")
        bar.set_edgecolor("black")

    fig.savefig(
        "./results/plots/uninterpretable_variance_explained.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def main(dimension_mapping_path, dimension_rating_path):
    dimension_mapping = load_dimension_mapping(dimension_mapping_path)
    ratings = loadmat(dimension_rating_path)["ratings"]
    df = process_dimension_ratings(ratings, dimension_mapping)
    df.to_csv("./data/misc/dimension_ratings_processed.csv", index=False)

    print_percentage_concept(df)
    print_percentage_quality(df)
    plot_quality(df)
    plot_concept(df)
    plot_concept_bar(df)

    # i think this is just a filtered version of the plot quality function. merge!
    plot_semantic_visual(df)

    embedding_paths = [
        ("Human", "./data/embeddings/human_behavior/parameters.npz"),
        ("VGG-16", "./data/embeddings/vgg16_bn/classifier.3/parameters.npz"),
        ("CLIP", "./data/embeddings/OpenCLIP/visual/parameters.npz"),
        ("Densenet", "./data/embeddings/densenet/global_pool/parameters.npz"),
        ("Resnet50", "./data/embeddings/resnet50/avgpool/parameters.npz"),
        ("Barlow-Twins", "./data/embeddings/barlowtwins-rn50/avgpool/parameters.npz"),
    ]

    embeddings = {name: load_sparse_codes(path) for name, path in embedding_paths}

    # TODO separate the plotting function here
    weighted_uninterpretability(embeddings, df)


if __name__ == "__main__":
    args = parse_args()
    main(args.dimension_mapping_path, args.dimension_rating_path)
