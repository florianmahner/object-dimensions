import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from PIL import Image


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

    ctr = 0
    save_path = os.path.join(plot_dir, key)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for triplet, dim_h, dim_d in zip(triplets, dims_human, dims_dnn):
        ctr += 1
        # fig = plot_figure(
        #     image_filenames,
        #     triplet,
        #     human_weights,
        #     dnn_weights,
        #     dim_h,
        #     dim_d,
        #     human_title,
        #     dnn_title,
        # )
        # fig.savefig(
        #     os.path.join(save_path, softmax_key + "_" + str(ctr) + ".pdf"),
        #     dpi=100,
        # )

        # plt.close(fig)

        if ctr == max:
            break

        # Plot the rose plots
        plot_bar(diff_dnn[ctr], os.path.join(save_path, f"rose_dnn_{str(ctr)}.pdf"))
        plot_bar(diff_human[ctr], os.path.join(save_path, f"rose_human_{str(ctr)}.pdf"))


def plot_bar(decisions, out_path="./rose.png"):
    # Get the topk most important dimensions that changed the most for a triplet

    dimensions = np.arange(1, len(decisions) + 1)
    topk = 5
    argsort = np.argsort(-decisions)
    decisions = decisions[argsort][:topk]

    # Check that the decision larger than eps
    eps = 1e-12
    indices = np.where(decisions > eps)[0]
    decisions = decisions[indices]

    # Filter the dimensions with the decisions larger than eps
    dimensions = dimensions[argsort][:topk]
    dimensions = dimensions[indices]
    decisions = decisions / decisions.sum()
    decisions = decisions * 1000
    decisions = np.array(decisions, dtype=int)

    dim = [f"Dim {str(i)}" for i in dimensions]
    df = pd.DataFrame({"softmax": decisions, "dim": dim})

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    plt.axis("off")

    lowerLimit = 0
    palette = sns.color_palette("husl", 13)

    # Compute the angle each bar is centered on:
    indexes = list(range(1, len(df.index) + 1))
    width2 = 2 * np.pi / len(df.index)
    angles = [element * width2 for element in indexes]

    # Draw bars
    bars = ax.bar(
        x=angles,
        height=df["softmax"],
        width=0.2,
        bottom=lowerLimit,
        linewidth=2,
        edgecolor="white",
        color=palette,
    )

    # little space between the bar and the label
    labelPadding = 4

    # Add labels
    for bar, angle, height, label in zip(bars, angles, df["softmax"], df["dim"]):
        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)

        # Flip some labels upside down
        alignment = ""
        if angle >= np.pi / 2 and angle < 3 * np.pi / 2:
            alignment = "right"
            rotation = rotation + 180
        else:
            alignment = "left"

        # Add a label to the bar right below the bar rotated
        ax.text(
            x=angle,
            y=height + labelPadding,
            s=label,
            rotation=rotation,
            rotation_mode="anchor",
            ha=alignment,
            va="center_baseline",
            fontsize=12,
        )

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)
