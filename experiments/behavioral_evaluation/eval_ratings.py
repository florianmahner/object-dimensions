import os
import pandas as pd
import glob
import numpy as np
import seaborn as sns
from textblob import TextBlob, Word
from itertools import chain
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict
import hashlib


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
    # TODO Replace Nones
    return df


def parse_csv(fpath: str) -> pd.DataFrame:
    df = pd.read_csv(fpath)
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


def make_wordclouds(df: pd.DataFrame) -> plt.Figure:
    max_words = 8
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
        return word_to_color.get(word, "black")

    wc = WordCloud(
        width=1600,
        height=800,
        background_color="white",
        color_func=custom_color_func,
        max_words=max_words,
        min_font_size=10,
    )

    wc = wc.generate_from_frequencies(words)
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def main() -> None:
    fpaths = glob.glob("./behavior-results/*.csv")
    assert len(fpaths) > 0, "No csv files found"
    out_path = "./results/dnn/variational/vgg16_bn/classifier.3/20.mio/sslab/150/256/1.0/14/analyses/behavior_wordclouds"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    dataframes = []
    for fpath in fpaths:
        print("Parsing: ", fpath)
        df = parse_csv(fpath)
        dataframes.append(df)

    df = compute_statistics(*dataframes)

    # Make a dataframe out of most common words and average interpretability
    df_save = df[["dim", "avg_interpretability", "most_common_words"]]

    # Save dataframe
    df_save.to_csv(
        os.path.join(os.path.dirname(out_path), "behavior_ratings_processed.csv"),
        index=False,
    )

    out_path_wc = "./results/dnn/variational/vgg16_bn/classifier.3/20.mio/sslab/150/256/1.0/14/analyses/"

    # new_path = os.path.join(out_path_wc, "per_dim", "**", "*10x50*")

    # files = glob.glob(new_path, recursive=True)

    # for f in files:
    #     os.remove(f)

    # Iterate over rows of df
    for dim in df["dim"]:
        print("Making wordcloud for dimension: ", dim, "...", end="\r")
        dim_df = df[df["dim"] == dim]
        fig = make_wordclouds(dim_df)
        dim_str = str(dim).zfill(2)
        for ext in ["pdf"]:
            plt.savefig(
                os.path.join(
                    out_path_wc,
                    "per_dim",
                    dim_str,
                    f"{str(dim)}_word_cloud_human.{ext}",
                ),
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )

        # fig.savefig(
        #     os.path.join(out_path, f"word_cloud{dim}.pdf"),
        #     bbox_inches="tight",
        #     pad_inches=0,
        #     dpi=300,
        # )
        plt.close(fig)


if __name__ == "__main__":
    main()
