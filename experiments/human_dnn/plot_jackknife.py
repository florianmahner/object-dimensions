import os
import toml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.gridspec as gridspec

from scipy.io import loadmat
from matplotlib.colors import ListedColormap
import matplotlib as mpl

import pandas as pd
import skimage.io as io


def plot_figure(
    image_filenames, triplet, human_weights, dnn_weights, dim_h, dim_d, title_1, title_2
):
    """Plot figure for jackknife analysis as a grid of the triplet and the topk images in a dimension"""
    fig = plt.figure(figsize=(18, 12))
    gs0 = gridspec.GridSpec(1, 3, figure=fig)
    gs0.update(top=0.95, bottom=0.7, left=0.25, right=0.75, wspace=0.0, hspace=0.0)
    lookup = ["i", "j", "k"]
    for ctr, img_index in enumerate(triplet):
        ax = plt.subplot(gs0[ctr])
        img = io.imread(image_filenames[img_index])
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
        # img = Image.open(image_filenames[human_idx])
        img = io.imread(image_filenames[human_idx])
        # img = img.resize((224, 224), Image.Resampling.BILINEAR)
        ax.imshow(img)
        ax.axis("off")

    plt.subplot(gs1[1]).set_title(title_1, fontsize=20)

    gs2 = gridspec.GridSpec(2, 3, figure=fig)
    gs2.update(top=0.65, bottom=0.1, left=0.55, right=0.95, wspace=0.0, hspace=0.0)

    for ctr, dnn_idx in enumerate(dnn):
        ax = plt.subplot(gs2[ctr])
        img = io.imread(image_filenames[dnn_idx])
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


def get_dominant_row_indices(matrix, top=2, threshold=0.50):
    """Get the indices of the rows that are dominant in the matrix, e.g. where the
    top elements of the row sum to 0.X of total sum"""
    dominant_indices = set()
    for i, row in enumerate(matrix):
        row = np.abs(row)
        sorted_dims = np.argsort(-row)
        row = row[sorted_dims]
        sum_dims = np.sum(row)
        sum_top = np.sum(row[:top])
        # I want that they explain less than the threshold in terms of variance, e.g. that they
        # are not sparse
        if sum_top / sum_dims <= threshold:
            dominant_indices.add(i)

    return dominant_indices


def plot_grid(
    plot_dir,
    image_filenames,
    jackknife_dict,
    key="dnn_k_human_k",
    softmax_key="high_both",
    max_topk=5,
    start_index=0,
):
    human_weights = jackknife_dict["human_weights"]
    dnn_weights = jackknife_dict["dnn_weights"]
    dnn_weights_plus = jackknife_dict["dnn_weights_plus"]

    jackknife = jackknife_dict[key]
    value = jackknife[softmax_key]
    triplets = value["triplets"][start_index:]
    dims_human = value["dims_human"][start_index:]
    dims_dnn = value["dims_dnn"][start_index:]
    diff_human = value["softmax_diff_human"][start_index:]
    diff_dnn = value["softmax_diff_dnn"][start_index:]

    # Here we filter non sparse triplets!
    # TODO Start of here again! NOTE I saw some images twice in the fucking plot of dimension visualization
    # No idea why, check code again based on what triplet I filter and what triplets I visualize later!
    # Also need to check if the dimensions of the DNN plus images actually match?? It seems like I use the
    # DNN behavior images for the comparison, but then try to visualize it for the plus...
    non_sparse_dnn = get_dominant_row_indices(diff_dnn, top=3, threshold=0.25)
    non_sparse_human = get_dominant_row_indices(diff_human, top=3, threshold=0.25)
    intersection_dominant_indices = list(non_sparse_dnn.intersection(non_sparse_human))
    diff_human = diff_human[intersection_dominant_indices, :]
    diff_dnn = diff_dnn[intersection_dominant_indices, :]
    dims_dnn = dims_dnn[intersection_dominant_indices]
    dims_human = dims_human[intersection_dominant_indices]
    triplets = triplets[intersection_dominant_indices]
    n_start = 0
    n_end = 200
    diff_human = diff_human[n_start:n_end]
    diff_dnn = diff_dnn[n_start:n_end]
    dims_dnn = dims_dnn[n_start:n_end]
    dims_human = dims_human[n_start:n_end]

    dnn_title = "DNN - " + key[-1]
    human_title = "Human - " + key.split("_")[1]

    plot_softmax_histogram(jackknife_dict, plot_dir)
    plot_weights(jackknife_dict, plot_dir)

    save_path = os.path.join(plot_dir, key)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Argsort diff human and diff dnn along axis 1
    topk_human = np.argsort(-diff_human, axis=1)[:, :max_topk]
    topk_human = {str(i): list(a) for i, a in enumerate(topk_human)}

    topk_dnn = np.argsort(-diff_dnn, axis=1)[:, :max_topk]
    topk_dnn = {str(i): list(a) for i, a in enumerate(topk_dnn)}
    save = {"human": topk_human, "dnn": topk_dnn}

    with open(os.path.join(save_path, "topk.toml"), "w") as f:
        toml.dump(save, f)

    for ctr, (triplet, dim_h, dim_d) in enumerate(zip(triplets, dims_human, dims_dnn)):
        fig = plot_figure(
            image_filenames,
            triplet,
            human_weights,
            dnn_weights_plus,
            dim_h,
            dim_d,
            human_title,
            dnn_title,
        )

        if ctr in [143, 60, 67, 37, 28]:
            dpi = 450
        else:
            dpi = 50

        fig.savefig(
            os.path.join(save_path, f"{str(ctr)}_" + softmax_key + ".pdf"),
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
            dpi=dpi,
            compression="tiff_lzw",
        )

        plt.close(fig)

        # Plot the rose plots
        plot_bar(diff_human[ctr], diff_dnn[ctr], ctr, save_path)


def get_66dim_cmap(dimcol_mat="./data/misc/colors66.mat"):
    dimcol_data = loadmat(dimcol_mat, simplify_cells=True)
    arr = dimcol_data["colors66"]
    dimcol_cmap = ListedColormap(arr, name="Behavioral dimensions", N=68)
    return dimcol_cmap


def savefig(fig, path):
    fig.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
        dpi=300,
        compression="tiff_lzw",
    )


def plot_bar_1d(human, dnn, ctr, out_path):
    sns.set(style="whitegrid", context="paper")
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    n = len(human)
    sns.barplot(x=np.arange(n), y=sorted(human, reverse=True), ax=ax, color="black")
    ax.set_xlabel("")
    ax.set_xticklabels([])
    sns.despine(offset=5)

    path_human = os.path.join(out_path, f"bar_human_{ctr}.pdf")
    savefig(fig, path_human)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    n = len(dnn)
    sns.barplot(x=np.arange(n), y=sorted(dnn, reverse=True), ax=ax, color="black")
    path_dnn = os.path.join(
        out_path,
        f"bar_dnn_{ctr}.pdf",
    )
    ax.set_xlabel("")
    ax.set_xticklabels([])
    sns.despine(offset=5)

    savefig(fig, path_dnn)


def plot_rose(softmax, out_path):
    dimcol_cmap = get_66dim_cmap()
    dimcol_cmap_hex = [mpl.colors.to_hex(row) for row in dimcol_cmap.colors]

    n = softmax.shape[0]
    dims = np.arange(n)

    # Save the colors with the softmax values sorted by the softmax values to json in outpath
    sort_index = np.argsort(-softmax)
    softmax_sorted = softmax[sort_index]
    dims_sorted = dims[sort_index]
    # with open(out_path.replace(".pdf", ".json"), "w") as f:
    #     json.dump(
    #         {
    #             "softmax_sorted": softmax_sorted.tolist(),
    #             "dims_sorted": dims_sorted.tolist(),
    #         },
    #         f,
    #     )

    df = pd.DataFrame({"softmax": softmax, "dims": dims})

    # max_softmax = np.max(df["softmax"])
    # hex_gray = "#cdcdcd"
    # dimcol_cmap_hex.append(hex_gray)  # Add gray color to the end of your color map
    # gray_color_index = (
    #     len(dimcol_cmap_hex) - 1
    # )  # Index of gray color in the extended cmap

    # # Update 'dims' in your DataFrame for gray color mapping
    # df.loc[df["softmax"] <= max_softmax / 10, "dims"] = gray_color_index
    # df.loc[df["softmax"] <= max_softmax / 10, "softmax"] = max_softmax / 10

    # # Plot with the updated 'dims'
    # fig = px.bar_polar(
    #     df,
    #     r="softmax",
    #     color="dims",
    #     color_discrete_sequence=dimcol_cmap_hex,  # Use your color map
    #     template="simple_white",
    # )

    # # # # Add shade of gray

    # max_softmax = np.max(softmax)
    # hex_gray = "#cdcdcd"
    # dimcol_cmap_hex.append(hex_gray)
    # ncolors = len(dimcol_cmap_hex)

    # df.loc[df["softmax"] <= max_softmax / 10, "dims"] = ncolors - 1
    # df.loc[df["softmax"] <= max_softmax / 10, "softmax"] = max_softmax / 10

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
        os.path.join(out_path, f"{ctr}_histogram.pdf"),
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)

    # Plot two rose plots
    for decisions, path in zip(
        [softmax_human, softmax_dnn],
        [f"{ctr}_rose_human.pdf", f"{ctr}_rose_dnn.pdf"],
    ):
        f_path = os.path.join(out_path, path)
        plot_rose(decisions, f_path)
