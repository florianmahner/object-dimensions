# %%
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from experiments import utils
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# %%
path = "/LOCAL/fmahner/object-dimensions/data/misc/dimlabels_human_wordclouds.tsv"
df = pd.read_csv(path, sep="\t")

path_vice = "/LOCAL/fmahner/object-dimensions/results/behavior/variational/4.12mio/sslab/150/256/1.0/21/params/parameters.npz"
path_spose = "/LOCAL/fmahner/object-dimensions/data/misc/spose_embedding_49d_sorted.txt"


weights_vice = utils.load_sparse_codes(path_vice)
weights_spose = utils.load_sparse_codes(path_spose)


def compare_modalities(weights_vice, weights_spose, duplicates=True):
    """Compare the Human behavior embedding to the VGG embedding by correlating them!"""
    assert (
        weights_spose.shape[0] == weights_vice.shape[0]
    ), "\nNumber of items in weight matrices must align.\n"

    dim_a = weights_spose.shape[1]
    dim_b = weights_vice.shape[1]

    corrs_between_modalities = np.zeros(dim_b)
    mod2_dims = []

    for dim_idx_1, weight_1 in enumerate(weights_vice.T):
        # Correlate modality 1 with all dimensions of modality 2
        corrs = np.zeros(dim_a)
        for dim_idx_2, weight_2 in enumerate(weights_spose.T):
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


corrs, mod2_dims = compare_modalities(weights_vice, weights_spose, duplicates=True)


# %%


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


# %%

import os

path = "/LOCAL/fmahner/object-dimensions/results/behavior/variational/4.12mio/sslab/150/256/1.0/21/wordclouds_spose_match/"

os.makedirs(path, exist_ok=True)

for i, dim_match in enumerate(mod2_dims):

    fig = make_wordclouds(df, dim_match)

    fig.savefig(
        f"/LOCAL/fmahner/object-dimensions/results/behavior/variational/4.12mio/sslab/150/256/1.0/21/wordclouds_spose_match/wordcloud_{i}.pdf",
        dpi=450,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close(fig)

    print(f"Saved wordcloud for dimension {i}.")

    # %%
