import os
import pandas as pd
import glob
import numpy as np
from textblob import TextBlob, Word
from itertools import chain
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict


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


def compute_frequencies(x: List[str]) -> Dict[str, int]:
    """count words, ignore duplicates and return a dict"""
    counter = Counter(x)
    counter = {k: v for k, v in counter.items() if k.strip()}
    return counter


def compute_frequencies(*dataframes: pd.DataFrame) -> pd.DataFrame:
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
        lambda x: list(chain(*x)), axis=1
    )
    grouped = grouped.drop(columns=description_headers)

    # compute most common words
    grouped["most_common_words"] = grouped["descriptions"].apply(
        lambda x: compute_frequencies(x)
    )

    return grouped


def make_wordclouds(df: pd.DataFrame) -> plt.Figure:
    wc = WordCloud(width=1000, height=500)

    wc = wc.generate_from_frequencies(df["most_common_words"].iloc[0])
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(wc)
    plt.axis("off")
    return fig


def main() -> None:
    fpaths = glob.glob("./behavior-res/*.csv")
    assert len(fpaths) > 0, "No csv files found"
    out_path = "./results/dnn/variational/vgg16_bn/classifier.3/20.mio/sslab/150/256/1.0/14/analyses/behavior_wordclouds"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    dataframes = []
    for fpath in fpaths:
        print("Parsing: ", fpath)
        df = parse_csv(fpath)
        dataframes.append(df)

    df = compute_frequencies(*dataframes)

    # Iterate over rows of df
    for dim in df["dim"]:
        print("Making wordcloud for dimension: ", dim, "...", end="\r")
        dim_df = df[df["dim"] == dim]
        fig = make_wordclouds(dim_df)
        fig.savefig(os.path.join(out_path, f"word_cloud{dim}.png"))
        plt.close(fig)


if __name__ == "__main__":
    main()
