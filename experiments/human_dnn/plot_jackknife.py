import os
import toml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from PIL import Image

from scipy.io import loadmat
from matplotlib.colors import ListedColormap
import matplotlib as mpl

import plotly.express as px
import pandas as pd


def plot_figure(
    image_filenames, triplet, human_weights, dnn_weights, dim_h, dim_d, title_1, title_2
):
    """Plot figure for jackknife analysis as a grid of the triplet and the topk images in a dimension"""
    fig = plt.figure(figsize=(12, 8))
    gs0 = gridspec.GridSpec(1, 3, figure=fig)
    gs0.update(top=0.95, bottom=0.7, left=0.25, right=0.75, wspace=0.0, hspace=0.0)
    lookup = ["i", "j", "k"]
    for ctr, img_index in enumerate(triplet):
        ax = plt.subplot(gs0[ctr])
        img = Image.open(image_filenames[img_index])
        img = img.resize((224, 224))

        ax.imshow(img)

        ax.set_ylabel("")
        ax.set_xlabel(lookup[ctr], fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplot(gs0[1]).set_title("Triplet", fontsize=20)

    human = human_weights[:, dim_h]
    human = np.argsort(-human)[:6]

    dnn = dnn_weights[:, dim_d]
    dnn = np.argsort(-dnn)[:6]
    # fig = plt.figure(figsize=(15, 5))
    gs1 = gridspec.GridSpec(2, 3, figure=fig)
    gs1.update(top=0.65, bottom=0.1, left=0.05, right=0.45, wspace=0.0, hspace=0.0)

    for ctr, human_idx in enumerate(human):
        ax = plt.subplot(gs1[ctr])
        img = Image.open(image_filenames[human_idx])
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        ax.imshow(img)
        ax.axis("off")

    plt.subplot(gs1[1]).set_title(title_1, fontsize=20)

    gs2 = gridspec.GridSpec(2, 3, figure=fig)
    gs2.update(top=0.65, bottom=0.1, left=0.55, right=0.95, wspace=0.0, hspace=0.0)

    for ctr, dnn_idx in enumerate(dnn):
        ax = plt.subplot(gs2[ctr])
        img = Image.open(image_filenames[dnn_idx])
        img = img.resize((224, 224))
        ax.imshow(img)
        ax.axis("off")
    plt.subplot(gs2[1]).set_title(title_2, fontsize=20)
    return fig


def plot_softmax_histogram(jackknife_dict, save_path):
    softmax_human = jackknife_dict["softmax_human"]
    softmax_dnn = jackknife_dict["softmax_dnn"]
    lookup = lambda x: "k" if x == 0 else "j" if x == 1 else "i"
    data = []
    for index in range(3):
        for softmax_val in softmax_human[:, index]:
            data.append(
                {
                    "Softmax": softmax_val,
                    "Type": "Human",
                    "Category": "OOO - " + lookup(index),
                }
            )
        for softmax_val in softmax_dnn[:, index]:
            data.append(
                {
                    "Softmax": softmax_val,
                    "Type": "DNN",
                    "Category": "OOO - " + lookup(index),
                }
            )

    df = pd.DataFrame(data)
    g = sns.FacetGrid(
        df, col="Type", row="Category", sharey=False, sharex=True, aspect=1.8
    )
    # g.map(sns.histplot, "Softmax", bins=100, kde=True)
    g.map_dataframe(
        lambda data, **kwargs: sns.histplot(
            data=data,
            x="Softmax",
            bins=100,
            kde=False,
            color="blue" if data["Type"].iloc[0] == "Human" else "red",
        )
    )
    g.set_axis_labels("Softmax", "Count")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.fig.subplots_adjust(top=0.9)

    g.savefig(
        os.path.join(save_path, "histogram_argmax.pdf"),
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(g.fig)


def plot_weights(jackknife_dict, save_path):
    weights_human = jackknife_dict["human_weights"]
    weights_dnn = jackknife_dict["dnn_weights"]

    weights_human = weights_human[weights_human != 0]
    weights_dnn = weights_dnn[weights_dnn != 0]

    # Set the style and context of the seaborn plots
    sns.set(style="whitegrid", context="paper")
    # Do a histogram of the weights using seaborn
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    sns.histplot(weights_human.flatten(), bins=100, kde=False, ax=ax[0], color="blue")
    ax[0].set_title("Human")
    sns.histplot(weights_dnn.flatten(), bins=100, kde=False, ax=ax[1], color="red")
    ax[1].set_title("DNN")
    ax[0].set_xlabel("Weight")
    ax[1].set_xlabel("Weight")
    ax[0].set_ylabel("Count")
    # Save the seaborn plot as a PDF
    fig.tight_layout()
    fig.savefig(
        os.path.join(save_path, "histogram_weights.pdf"),
        bbox_inches="tight",
        transparent=True,
    )

    plt.close(fig)


def plot_grid(
    plot_dir,
    image_filenames,
    jackknife_dict,
    key="dnn_k_human_k",
    softmax_key="high_both",
    max=5,
    start_index=0,
):
    human_weights = jackknife_dict["human_weights"]
    dnn_weights = jackknife_dict["dnn_weights"]
    jackknife = jackknife_dict[key]
    value = jackknife[softmax_key]
    triplets = value["triplets"][start_index:]
    dims_human = value["dims_human"][start_index:]
    dims_dnn = value["dims_dnn"][start_index:]
    diff_human = value["softmax_diff_human"][start_index:]
    diff_dnn = value["softmax_diff_dnn"][start_index:]

    human_title = "Human - " + key[-1]
    dnn_title = "DNN - " + key.split("_")[1]

    plot_softmax_histogram(jackknife_dict, plot_dir)
    plot_weights(jackknife_dict, plot_dir)

    save_path = os.path.join(plot_dir, key)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Argsort diff human and diff dnn along axis 1
    topk_human = np.argsort(-diff_human, axis=1)[:, :5]
    topk_human = {str(i): list(a) for i, a in enumerate(topk_human)}

    topk_dnn = np.argsort(-diff_dnn, axis=0)[:, :5]
    topk_dnn = {str(i): list(a) for i, a in enumerate(topk_dnn)}
    save = {"human": topk_human, "dnn": topk_dnn}
    with open(os.path.join(save_path, "topk.toml"), "w") as f:
        toml.dump(save, f)

    for ctr, (triplet, dim_h, dim_d) in enumerate(zip(triplets, dims_human, dims_dnn)):
        fig = plot_figure(
            image_filenames,
            triplet,
            human_weights,
            dnn_weights,
            dim_h,
            dim_d,
            human_title,
            dnn_title,
        )
        fig.savefig(
            os.path.join(save_path, softmax_key + "_" + f"{str(ctr + 1)}" + ".pdf"),
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
        )

        plt.close(fig)

        # Plot the rose plots
        plot_bar(diff_human[ctr], diff_dnn[ctr], ctr, save_path)


def get_66dim_cmap(dimcol_mat="./data/misc/colors66.mat"):
    dimcol_data = loadmat(dimcol_mat, simplify_cells=True)
    arr = dimcol_data["colors66"]
    dimcol_cmap = ListedColormap(arr, name="Behavioral dimensions", N=68)
    return dimcol_cmap


def plot_plotly(softmax, out_path):
    dimcol_cmap = get_66dim_cmap()
    dimcol_cmap_hex = [mpl.colors.to_hex(row) for row in dimcol_cmap.colors]

    n = softmax.shape[0]
    dims = np.arange(n)

    df = pd.DataFrame({"softmax": softmax, "dims": dims})

    fig = px.bar_polar(
        df,
        r="softmax",
        color="dims",
        color_continuous_scale=dimcol_cmap_hex,
        template="simple_white",
    )
    # hide gridlines, labels, and ticks.
    fig.update_layout(
        polar=dict(
            radialaxis=dict(showticklabels=False, ticks=""),
            angularaxis=dict(showticklabels=False, ticks=""),
        )
    )
    fig.update_polars(
        radialaxis_showline=False,
        angularaxis_showline=False,
    )
    fig.write_image(out_path, width=800, height=800)


def plot_bar(softmax_human, softmax_dnn, ctr, out_path="./rose.png"):
    sns.set(style="whitegrid", context="paper")
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    sns.histplot(softmax_human.flatten(), bins=100, kde=False, ax=ax[0], color="blue")
    ax[0].set_title("Human")
    sns.histplot(softmax_dnn.flatten(), bins=100, kde=False, ax=ax[1], color="red")
    ax[1].set_title("DNN")
    ax[0].set_xlabel("Δ Softmax")
    ax[1].set_xlabel("Δ Softmax")
    ax[0].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_path, f"histogram_{ctr + 1}.pdf"),
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)

    # Plot two rose plots
    for decisions, path in zip(
        [softmax_human, softmax_dnn],
        [f"rose_human_{ctr + 1}.pdf", f"rose_dnn_{ctr + 1}.pdf"],
    ):
        f_path = os.path.join(out_path, path)
        plot_plotly(decisions, f_path)
