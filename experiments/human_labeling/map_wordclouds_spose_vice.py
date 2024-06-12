""" This script takes the human annoted dimension labels obtained from a non-bayesian variant of our model (SPoSE) and uses these
to label each dimension in our bayesian model (VICE). Both embeddings have been trained on the same data and highly correlate, 
so we can use the human labels to assign the most likely human label to each dimension in the VICE embedding. We then generate
wordclouds for each dimension in the VICE embedding, where the size of the words corresponds to the frequency of the word in the
human annotations"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from objdim.utils import load_sparse_codes
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os


def pairiwise_correlate_dimensions(
    vice_embedding,
    spose_embedding,
    duplicates=True,
):
    """Compare the Human behavior embedding to the VGG embedding by correlating them!"""

    num_vice, dims_vice = vice_embedding.shape
    num_spose, dims_spose = spose_embedding.shape
    assert num_spose == num_vice, "\nNumber of items in weight matrices must align.\n"
    corrs_between_modalities = np.zeros(dims_vice)
    mod2_dims = []

    for dim_idx_1, weight_1 in enumerate(vice_embedding.T):
        # Correlate modality 1 with all dimensions of modality 2
        corrs = np.zeros(dims_spose)
        for dim_idx_2, weight_2 in enumerate(spose_embedding.T):
            corrs[dim_idx_2] = pearsonr(weight_1, weight_2)[0]

        if duplicates:
            mod2_dims.append(np.argmax(corrs))

        else:
            for ind_mod_2 in np.argsort(-corrs):
                if ind_mod_2 not in mod2_dims:
                    mod2_dims.append(ind_mod_2)
                    break

        corrs_between_modalities[dim_idx_1] = corrs[mod2_dims[-1]]

    return corrs_between_modalities, mod2_dims


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


def make_wordclouds(df: pd.DataFrame, dim: int) -> plt.Figure:
    max_words = 6
    colors = sns.color_palette("deep")[:max_words]
    colors = rgb_to_hex(colors)

    # Filter dataframe by dimension
    filtered_df = df[df["dim"] == dim]

    # Generate word frequencies
    words = filtered_df.set_index("label")["count"].to_dict()

    # sort words by frequency
    top_words = {
        k: v for k, v in sorted(words.items(), key=lambda item: item[1], reverse=True)
    }

    sorted_words = list(top_words.keys())[:max_words]

    adjusted_words = {
        word: (freq if freq != 1 else 0.001) for word, freq in words.items()
    }

    # Map each word to a color based on its index
    word_to_color = {
        word: colors[i % len(colors)] for i, word in enumerate(sorted_words)
    }

    def custom_color_func(
        word, font_size, position, orientation, random_state=None, **kwargs
    ):
        if words[word] == 1:
            return "gray"
        return word_to_color.get(word, "black")

    diameter = 1000
    mask = create_circular_mask(diameter)

    import matplotlib.font_manager as fm

    arial_font_path = fm.findfont(fm.FontProperties(family="Arial"))

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
        max_font_size=300,
    )

    wc = wc.generate_from_frequencies(adjusted_words)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


def main():
    path = "./data/misc/spose_dimlabels_human_wordclouds.tsv"
    df = pd.read_csv(path, sep="\t")

    path_vice = "./data/embeddings/human/parameters.npz"
    path_spose = "./data/embeddings/misc/spose_embedding_49d_sorted.txt"

    embedding_vice = load_sparse_codes(path_vice)
    embedding_spose = load_sparse_codes(path_spose)

    corrs, mod2_dims = pairiwise_correlate_dimensions(
        embedding_vice, embedding_spose, duplicates=True
    )

    path = "./results/wordclouds_spose_match/"
    os.makedirs(path, exist_ok=True)

    for i, dim_match in enumerate(mod2_dims):
        fig = make_wordclouds(df, dim_match)

        fig.savefig(
            os.path.join(path, f"wordcloud_{i}.pdf"),
            dpi=450,
            bbox_inches="tight",
            pad_inches=0,
        )

        print(f"Saved wordcloud for dimension {i}.")
        plt.close(fig)


if __name__ == "__main__":
    main()
