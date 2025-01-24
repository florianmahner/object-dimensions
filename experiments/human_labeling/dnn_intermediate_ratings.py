""" This file analyzes the dimension ratings provided by human annotators to label the dimensions of the DNN and human embeddings"""

import json
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from objdim.utils import load_sparse_codes
from tomlparse import argparse

# Identifiers for the models in the ratings
LAYER_TO_STAGE = {
    "features.5": "early",
    "features.22": "middle",
    "features.42": "late",
    "classifier.4": "penultimate",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dimension_rating_csv",
        type=str,
        default="./data/misc/early_middle_late_ratings/ratings_corrected_3.csv",
        help="Path to the mat file containing the human ratings",
    )
    return parser.parse_args()


def plot_quality(df):
    df = df[df["Quality"] != "unclear"]

    quality_counts = (
        df.groupby(["Layer", "Quality"], observed=False).size().unstack(fill_value=0)
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
            y="Layer",
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

    ax.set(xlabel="Percentage of Ratings", ylabel="Layer")
    sns.despine(left=True, bottom=True)
    fig.savefig(
        "./results/plots/early_middle_late_quality_ratings.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def plot_concept(df):
    # Counting occurrences of each concept per model and convert to percentage
    concept_counts = (
        df.groupby(["Layer", "Concept"], observed=False).size().unstack(fill_value=0)
    )
    concept_percent = concept_counts.div(
        concept_counts.sum(axis=1), axis=0
    ).reset_index()

    # Plotting the stacked bar plot for concept ratings using seaborn and matplotlib
    fig, ax = plt.subplots(figsize=(6, 4))

    # Initialize the left position for stacking
    left = pd.Series([0] * len(concept_percent), index=concept_percent.index)

    # List of concepts for consistent ordering
    concepts_list = ["single concept", "multiple concepts"]

    # Default seaborn color palette for concepts
    palette_concepts = sns.color_palette("deep")

    # Map specific colors to each concept using default palette
    concept_to_color = {
        "single concept": palette_concepts[0],  # blue
        "multiple concepts": palette_concepts[1],  # green
    }

    # Plot each concept as a separate bar
    for concept in concepts_list:
        sns.barplot(
            y="Layer",
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
    ax.set(xlabel="Fraction of Ratings", ylabel="Layer")
    sns.despine(left=True, bottom=True)
    fig.savefig(
        "./results/plots/early_middle_late_concept_ratings.pdf",
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


def print_percentage_concept(df):
    for layer in df["Layer"].unique():
        for concept in df["Concept"].unique():
            layer_df = df[df["Layer"] == layer]
            concept_df = layer_df[layer_df["Concept"] == concept]
            percentage = len(concept_df) / len(layer_df) * 100
            print(f"Percentage of {concept} ratings in {layer}: {percentage:.2f}%")


def print_percentage_quality(df):
    for layer in df["Layer"].unique():
        for quality in df["Quality"].unique():
            layer_df = df[df["Layer"] == layer]
            quality_df = layer_df[layer_df["Quality"] == quality]
            percentage = len(quality_df) / len(layer_df) * 100
            print(f"Percentage of {quality} ratings in {layer}: {percentage:.2f}%")


def plot_semantic_visual(df):
    # Filter for early, middle, and late stages
    df_filtered = df[df["Layer"].isin(LAYER_TO_STAGE.keys())]

    # Map layer names to stages
    df_filtered["Layer"] = df_filtered["Layer"].map(LAYER_TO_STAGE)

    # Remove 'unclear' quality ratings
    df_filtered = df_filtered[df_filtered["Quality"] != "unclear"]

    # Count occurrences of each quality and convert to percentages
    percentage_df = (
        df_filtered.groupby(["Layer", "Quality"]).size().unstack(fill_value=0)
    )
    percentage_df = percentage_df.div(percentage_df.sum(axis=1), axis=0).reset_index()
    percentage_df = percentage_df.melt(
        id_vars=["Layer"], var_name="Quality", value_name="Percentage"
    )

    sns.set_context("paper")

    palette_qualities = sns.color_palette("deep")

    # Plotting the stacked bar plot using seaborn and matplotlib
    fig, ax = plt.subplots(figsize=(6, 4))

    # List of qualities for consistent ordering
    qualities_list = ["semantic", "mix visual-semantic", "visual"]

    # Map specific colors to each quality
    quality_to_color = {
        "semantic": palette_qualities[0],
        "visual": palette_qualities[2],
        "mix visual-semantic": palette_qualities[1],
    }

    # Create a pivot table for easier plotting
    percentage_pivot = percentage_df.pivot(
        index="Layer", columns="Quality", values="Percentage"
    ).fillna(0)

    # Sort the index to have Early at the top, Middle in the middle, and Late at the bottom
    percentage_pivot = percentage_pivot.reindex(
        ["early", "middle", "late", "penultimate"]
    )

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

    ax.set(xlabel="Percentage of Ratings", ylabel="Layer")
    sns.despine(left=True, bottom=True)

    # Capitalize the y-axis labels
    ax.set_yticks(range(len(percentage_pivot.index)))
    ax.set_yticklabels(["Early", "Middle", "Late", "Penultimate"])

    fig.savefig(
        "./results/plots/early_middle_late_quality_ratings.pdf",
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
        layer = df[df["Layer"] == name]
        uninterpretable = layer[layer["Concept"] == "uninterpretable"]
        uninterpretable_variance = variance[uninterpretable["Dimension"]].sum()
        variances[name] = uninterpretable_variance * 100

    # here is the plotting part now.
    variance = pd.DataFrame(
        variances.items(), columns=["Layer", "Uninterpretable Variance"]
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x="Layer",
        y="Uninterpretable Variance",
        data=variance,
        ax=ax,
        edgecolor="black",
        facecolor=".7",
    )
    ax.set(
        xlabel="Layer",
        ylabel="Variance in Embedding Explained\nby Uninterpretable Dimensions [%]",
    )
    sns.despine(left=True, bottom=True)
    # make hatch
    # for i, bar in enumerate(ax.patches):
    #     bar.set_hatch("//")
    #     bar.set_edgecolor("black")

    fig.savefig(
        "./results/plots/early_middle_late_uninterpretable_variance.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def main(dimension_rating_csv):
    df_ratings = pd.read_csv(dimension_rating_csv)

    print_percentage_concept(df_ratings)
    print_percentage_quality(df_ratings)
    plot_quality(df_ratings)
    plot_concept(df_ratings)

    # i think this is just a filtered version of the plot quality function. merge!
    plot_semantic_visual(df_ratings)

    embedding_paths = [
        ("Early", "./data/embeddings/vgg16_bn/features.5/parameters.npz"),
        ("Middle", "./data/embeddings/vgg16_bn/features.22/parameters.npz"),
        ("Late", "./data/embeddings/vgg16_bn/features.42/parameters.npz"),
    ]

    embeddings = {name: load_sparse_codes(path) for name, path in embedding_paths}

    # TODO separate the plotting function here
    weighted_uninterpretability(embeddings, df_ratings)


if __name__ == "__main__":
    args = parse_args()
    main(args.dimension_rating_csv)
