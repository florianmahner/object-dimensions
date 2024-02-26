#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" This script generate GPT3 features for all images from the THINGS image dataset where behavioral data
has been colleted! For each of these images GPT3 features have been generated. This script correlates 
these features with the learned embedding and find the descriptions that best describe each dimension """

import os

import pandas as pd
import numpy as np
from copy import deepcopy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tomlparse import argparse
from object_dimensions.utils import load_sparse_codes, load_image_data
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(
        description="Label dimensions of sparse codes using GPT3"
    )
    parser.add_argument(
        "--embedding_path",
        default="./embedding.txt",
        type=str,
        help="Path to the embedding txt file",
    )
    parser.add_argument(
        "--img_root",
        default="./data/image_data/images12_plus",
        type=str,
        help="Path to all vgg features",
    )
    parser.add_argument(
        "--feature_norm_path",
        default="./data/feature_norms",
        type=str,
        help="Path to GPT3 norms",
    )
    parser.add_argument(
        "--run", default=False, action="store_true", help="Run extraction of norms"
    )
    return parser.parse_args()


def normalize_object_dimension_weights(object_dimension_embeddings):
    # Normalize object dimension weights
    dim_sums = object_dimension_embeddings.sum(axis=0)
    normalized_object_dimension_embeddings = object_dimension_embeddings.div(dim_sums)

    return normalized_object_dimension_embeddings


def compute_feature_weights_for_dims(
    object_feature_embeddings, object_dimension_embeddings, dims
):
    # Compute feature weight for each dimension
    weighted_features_for_all_dims = pd.DataFrame()
    for dim in dims:
        print(f"Run dimension: {dim}")
        df = deepcopy(object_feature_embeddings)
        dimension_values = object_dimension_embeddings.loc[:, dim]

        df_weighted_by_dim = df.mul(dimension_values, axis=0)
        df_summed_over_objects = df_weighted_by_dim.sum(axis=0).to_frame().sort_index()

        df_renamed = df_summed_over_objects.rename(
            columns={df_summed_over_objects.columns[0]: dim}
        )
        weighted_features_for_all_dims = pd.concat(
            [weighted_features_for_all_dims, df_renamed], axis=1
        )

    return weighted_features_for_all_dims


def normalize_feature_weights_across_dims(weighted_features_for_all_dims, dims):
    # Normalize feature weights across dimensions by substracting mean value based on all other dimensions
    normed_features = pd.DataFrame()
    for dim in dims:
        features_for_all_other_dims = weighted_features_for_all_dims.drop(dim, axis=1)
        mean_per_feature = features_for_all_other_dims.mean(axis=1)

        dim_values = weighted_features_for_all_dims.loc[:, dim]
        dim_values_normed = dim_values.subtract(mean_per_feature).to_frame()

        dim_values_normed = dim_values_normed.rename(
            columns={dim_values_normed.columns[0]: dim}
        )
        normed_features = pd.concat([normed_features, dim_values_normed], axis=1)

    return normed_features


def calc_top_feature_per_dim(
    object_dimension_embeddings, object_feature_embeddings, dims
):
    """
    Calculate the top feature for each dimension by weighting the feature frequency with the object dimension weights.
    :param object_feature_embeddings: is a objectXfeature matrix with either frequency or a normalized count like tfidf
    :param object_feature_embeddings: is a objectXdimension matrix with dimension weights for each object
    :param dims: is a list of dimension names
    """
    object_dimension_embeddings = normalize_object_dimension_weights(
        object_dimension_embeddings
    )
    weighted_features_for_all_dims = compute_feature_weights_for_dims(
        object_feature_embeddings, object_dimension_embeddings, dims
    )
    normed_features = normalize_feature_weights_across_dims(
        weighted_features_for_all_dims, dims
    )

    return normed_features


def matrix_to_top_list(df, topk=6):
    df_list = pd.DataFrame()
    for dim in df.columns:
        dim_values = df.loc[:, [dim]].reset_index()
        dim_values = dim_values.rename(
            columns={dim_values.columns[0]: "feature", dim_values.columns[1]: "weight"}
        )
        top = dim_values.sort_values(by="weight", ascending=False)[:topk]
        top["dimension"] = dim
        df_list = pd.concat([df_list, top])

    return df_list


def rgb_to_hex(rgb_values):
    hex_colors = []
    for rgb in rgb_values:
        r, g, b = [int(x * 255) for x in rgb]
        hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
        hex_colors.append(hex_color)
    return hex_colors


def generate_gpt3_norms(
    embedding_path,
    img_root,
    feature_norm_path="./data/feature_norms/feature_object_matrix.csv",
    run=False,
):
    # Save to file
    base_path = os.path.dirname(os.path.dirname(embedding_path))
    out_path = os.path.join(base_path, "analyses", "per_dim")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    fname = os.path.join(out_path, "dimensions_labelled_gpt3.csv")

    feature_norms = pd.read_csv(feature_norm_path)
    feature_norms = feature_norms.T

    embedding = load_sparse_codes(embedding_path)

    image_filenames, indices = load_image_data(img_root, filter_behavior=True)
    embedding = embedding[indices]

    objects = [os.path.basename(f).split(".")[0][:-4] for f in image_filenames]

    if run:
        embedding_pd = pd.DataFrame(embedding, index=objects)

        descriptions = calc_top_feature_per_dim(
            embedding_pd, feature_norms, list(range(embedding.shape[1]))
        )
        topk_descriptions = matrix_to_top_list(descriptions, topk=7)

        topk_indices = topk_descriptions["feature"].tolist()

        topk_indices = topk_descriptions["feature"].tolist()

        # Get the first row of the feature norms pd dataframe
        text = feature_norms.iloc[0, :].tolist()
        features = [text[t] for t in topk_indices]
        topk_descriptions["description"] = features
        topk_descriptions["description"] = features
        topk_descriptions.to_csv(fname, index=False)

    else:
        topk_descriptions = pd.read_csv(
            fname, dtype={"feature": str, "weight": float, "dimension": int}
        )

    # Create a word cloud for each dimension with its description weights by the weight
    for dim in range(embedding.shape[1]):
        dim_df = topk_descriptions[topk_descriptions["dimension"] == dim]
        dim_df = dim_df.sort_values(by="weight", ascending=False)
        words = dim_df["description"].tolist()
        weights = dim_df["weight"].tolist()
        dim_df = topk_descriptions[topk_descriptions["dimension"] == dim]
        dim_df = dim_df.sort_values(by="weight", ascending=False)
        words = dim_df["description"].tolist()
        weights = dim_df["weight"].tolist()
        weights = [w * 1000 for w in weights]
        weights = [int(w) for w in weights]

        frequencies = {k: v for k, v in zip(words, weights)}
        max_words = 6

        colors = sns.color_palette("deep")[:max_words]
        colors = rgb_to_hex(colors)

        # sort words by frequency
        top_frequencies = {
            k: v
            for k, v in sorted(
                frequencies.items(), key=lambda item: item[1], reverse=True
            )
        }

        sorted_words = list(top_frequencies.keys())[:max_words]

        # Map each word to a color based on its index
        word_to_color = {
            word: colors[i % len(colors)] for i, word in enumerate(sorted_words)
        }

        def custom_color_func(
            word, font_size, position, orientation, random_state=None, **kwargs
        ):
            return word_to_color.get(word, "black")

        wordcloud = WordCloud(
            width=1600,
            height=800,
            max_words=6,
            background_color="white",
            color_func=custom_color_func,
        ).generate_from_frequencies(dict(zip(words, weights)))
        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        for ext in ["pdf", "png"]:
            plt.savefig(
                os.path.join(out_path, f"{dim:02d}", f"{dim}_word_cloud_gpt.{ext}"),
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )
        plt.close()


if __name__ == "__main__":
    args = parse_args()
    generate_gpt3_norms(
        args.embedding_path, args.img_root, args.feature_norm_path, args.run
    )