import os
import pandas as pd
import glob
import numpy as np
import seaborn as sns
from textblob import Word
from itertools import chain
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict
import matplotlib.font_manager as fm
from tomlparse import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--behavior_ratings_path",
        type=str,
        default="./data/behavior_ratings/*.csv",
        help="Path to the csv files containing the human ratings",
    )
    return parser.parse_args()


def correct_spelling_errors(df: pd.DataFrame) -> pd.DataFrame:
    def certainty_correction(word):
        if word == "":
            return word
        processor = Word(word)
        correction, logit = processor.spellcheck()[0]
        if logit > 0.99:
            correction
        else:
            return word

    df = df.applymap(certainty_correction)
    return df


def parse_csv(fpath: str) -> pd.DataFrame:
    df = pd.read_csv(fpath)
    print(fpath)
    response_headers = [f"description_{i}" for i in range(1, 6)]
    responses = df[response_headers]
    responses = responses.fillna("")

    # Correct spelling errors
    # responses = correct_spelling_errors(responses)
    df[response_headers] = responses
    df = df.sort_values(by=["dim"])
    df.drop(columns=["image_file"], inplace=True)

    return df


def compute_word_frequencies(x: List[str]) -> Dict[str, int]:
    """count words, ignore duplicates and return a dict"""
    counter = Counter(x)
    counter = {k: v for k, v in counter.items() if k.strip()}
    return counter


def convert_to_list(x):
    """Remove empty string at beginning and end"""
    x = list(chain(*x))
    x = [i for i in x if i not in ["", "None"]]
    x = [i.strip() for i in x]
    return x


def compute_statistics(*dataframes: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat(dataframes, axis=0)
    # group responses based on dimension into a list
    grouped = df.groupby("dim").agg(list).reset_index()

    # compute average interpretability of column interpretability
    grouped["avg_interpretability"] = grouped["interpretability"].apply(
        lambda x: np.mean(x)
    )

    # get stats of avg_interpretability
    descriptive_interpretabilies = grouped["avg_interpretability"].describe()
    print(
        "\nDescriptive statistics of interpretability \n", descriptive_interpretabilies
    )
    description_headers = [f"description_{i}" for i in range(1, 6)]

    grouped["descriptions"] = grouped[description_headers].apply(
        convert_to_list, axis=1
    )

    grouped = grouped.drop(columns=description_headers)

    # compute most common words
    grouped["most_common_words"] = grouped["descriptions"].apply(
        lambda x: compute_word_frequencies(x)
    )

    return grouped


def rgb_to_hex(rgb_values):
    hex_colors = []
    for rgb in rgb_values:
        r, g, b = [int(x * 255) for x in rgb]
        hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
        hex_colors.append(hex_color)
    return hex_colors


def create_circular_mask(diameter):
    # Create a high-resolution circular mask
    x, y = np.ogrid[:diameter, :diameter]
    mask = (x - diameter // 2) ** 2 + (y - diameter // 2) ** 2 > (
        diameter // 2 - 20
    ) ** 2
    mask = 255 * mask.astype(int)
    return mask


def make_wordclouds(df: pd.DataFrame) -> plt.Figure:

    # Set the default font family to Arial in matplotlib
    plt.rcParams["font.family"] = "Arial"
    arial_font_path = fm.findfont(fm.FontProperties(family="Arial"))

    max_words = 6
    colors = sns.color_palette("deep")[:max_words]
    colors = rgb_to_hex(colors)

    words = df["most_common_words"].iloc[0]

    # Take all words. Alternative is filtering out words that have a frequency larger 0 (i.e. appear at least once)
    words = {k: v for k, v in words.items() if v > 0}

    # sort words by frequency
    top_words = {
        k: v for k, v in sorted(words.items(), key=lambda item: item[1], reverse=True)
    }

    sorted_words = list(top_words.keys())[:max_words]

    # Map each word to a color based on its index
    word_to_color = {
        word: colors[i % len(colors)] for i, word in enumerate(sorted_words)
    }

    def custom_color_func(
        word, font_size, position, orientation, random_state=None, **kwargs
    ):
        return word_to_color.get(word, "grey")

    diameter = 1000
    mask = create_circular_mask(diameter)

    wc = WordCloud(
        font_path=arial_font_path,
        width=diameter,
        height=diameter,
        background_color="white",
        color_func=custom_color_func,
        max_words=max_words,
        min_font_size=0.1,
        mask=mask,
        prefer_horizontal=1.0,
        scale=3,  # Increase the scale to reduce whitespace
        margin=0,  # Reduce the margin to minimize whitespace
    )

    wc = wc.generate_from_frequencies(words)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def parse_csv_into_df(filepaths: str) -> pd.DataFrame:
    dataframes = []
    for csv_file in filepaths:
        dataframes.append(pd.read_csv(csv_file))
    return dataframes


def main(behavior_ratings_path: str) -> None:
    filepaths = glob.glob(
        os.path.join(behavior_ratings_path, "**", ".csv"), recursive=True
    )
    assert (
        len(filepaths) > 0
    ), f"No csv files found in the specified path {behavior_ratings_path}"

    ratings = parse_csv_into_df(filepaths)
    ratings = compute_statistics(*ratings)

    # Make a dataframe out of most common words and average interpretability
    ratings_out = ratings[["dim", "avg_interpretability", "most_common_words"]]

    out_path = "./results/dimension_labeling"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    ratings_out.to_csv(
        os.path.join(out_path, "behavior_ratings_processed.csv"),
        index=False,
    )

    for dim in ratings["dim"]:
        print("Making wordcloud for dimension: ", dim, "...", end="\r")
        dim_df = ratings[ratings["dim"] == dim]
        fig = make_wordclouds(dim_df)
        dim_str = str(dim).zfill(2)
        save_path = os.path.join(out_path, f"{dim_str}_word_cloud_human.pdf")
        fig.savefig(save_path, dpi=450, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    main(args.behavior_ratings_path)
