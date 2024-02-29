import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", context="paper")

from PIL import Image
from object_dimensions.utils import (
    load_sparse_codes,
    load_image_data,
)
from object_dimensions.latent_predictor import LatentPredictor
from experiments.visualization import plot_dim_3x2
from tomlparse import argparse

import matplotlib

matplotlib.rcParams["font.sans-serif"] = "Arial"
matplotlib.rcParams["font.family"] = "sans-serif"


def parse_args():
    parser = argparse.ArgumentParser(description="Causal comparison")
    parser.add_argument(
        "--embedding_path",
        type=str,
        default="./results",
        help="Path to final embedding",
    )
    parser.add_argument(
        "--img_root",
        type=str,
        default="./data/things_plus",
        help="Path to image root directory",
    )
    return parser.parse_args()


# Each entry of dims is sorted by increase, decrease, relevant control, irrelevant control
CAUSAL_DIMS = dict()
CAUSAL_DIMS["toilet.jpg"] = {
    "increase": 25,
    "decrease": 10,
    "relevant_control": 13,
    "irrelevant_control": 8,
}

# CAUSAL_DIMS["chocolate.jpg"] = {
#     "increase": 37,
#     "decrease": 4,
#     "relevant_conrol": 12,
#     "irrelevant_control": 8,
# }

# CAUSAL_DIMS["calf1.jpg"] = {
#     "increase": 20,
#     "decrease": 56,
#     "relevant_control": 22,
#     "irrelevant_control": 62,
# }

CAUSAL_DIMS["lifesaver.jpg"] = {
    "increase": 28,
    "decrease": 33,
    "relevant_control": 26,
    "irrelevant_control": 19,
}

CAUSAL_DIMS["manhole.jpg"] = {
    "increase": 28,
    "decrease": 33,
    "relevant_control": 26,
    "irrelevant_control": 19,
}

CAUSAL_DIMS["bottle.jpg"] = {
    "increase": 39,
    "decrease": 51,
    "relevant_control": 35,
    "irrelevant_control": 19,
}

CAUSAL_DIMS["basketball.jpg"] = {
    "increase": 4,
    "decrease": 37,
    "relevant_control": 28,
    "irrelevant_control": 14,
}

CAUSAL_DIMS["footprint.jpg"] = {
    "increase": 20,
    "decrease": 56,
    "relevant_control": 22,
    "irrelevant_control": 62,
}


def load_img_and_predict_codes(path, predictor):
    # Load the image path using PIL and convert to torch tensor
    img = Image.open(path).convert("RGB")
    codes = predictor(img)[1]
    codes = codes.detach().numpy()
    return codes


def run_causal_comparison(embedding_path, img_root):
    base_path = os.path.dirname(os.path.dirname(embedding_path))
    regression_path = os.path.join(base_path, "analyses", "sparse_codes")
    predictor = LatentPredictor(regression_path=regression_path, device="cpu")

    causal_root = os.path.join("./data", "causal_images")
    out_path = os.path.join(base_path, "analyses", "causal_comparison")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    images, indices = load_image_data(img_root, filter_plus=True)
    Y = load_sparse_codes(embedding_path)
    Y = Y[indices]

    for img_name, dim_order in CAUSAL_DIMS.items():
        # Create path
        img_base = img_name.split(".")[0]
        f_path = os.path.join(out_path, img_base)
        if os.path.exists(f_path):
            for f in os.listdir(f_path):
                os.remove(os.path.join(f_path, f))
        else:
            os.makedirs(f_path)

        print("Running for ", img_name)
        orig_path = os.path.join(causal_root, "original", img_name)
        codes_orig = load_img_and_predict_codes(orig_path, predictor)
        manip_path = os.path.join(causal_root, "manipulated", img_name)
        codes_manip = load_img_and_predict_codes(manip_path, predictor)
        decrease = np.argsort(codes_orig - codes_manip)[:10]
        increase = np.argsort(codes_manip - codes_orig)[:10]
        original = np.argsort(-codes_orig)[:10]
        manipulated = np.argsort(-codes_manip)[:10]

        print("Dims important for original image: ", original[:10])
        print("Dims important for manipulated image: ", manipulated[:10])
        print("Dims with largest decrease", increase[:10])
        print("Dims with largest increase", decrease[:10])

        all_diffs = codes_manip - codes_orig
        # PLot the histogram of the differences
        fig = plt.figure(1, figsize=(6, 4))
        sns.set(font_scale=1.5)
        sns.set_style("whitegrid")
        ax = sns.barplot(
            x=np.arange(0, len(all_diffs), 1),
            y=all_diffs,
            edgecolor=".8",
            facecolor=(0, 0, 0, 0),
        )

        # colors = [
        #     "#1f77b4",
        #     "#ff7f0e",
        #     "#2ca02c",
        #     "#d62728",
        #     "#9467bd",
        #     "#8c564b",
        #     "#e377c2",
        #     "#7f7f7f",
        #     "#bcbd22",
        #     "#17becf",
        # ]

        colors = sns.color_palette("deep")

        # give colors in rgb 0-255
        colors_rgb = [
            (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)) for x in colors
        ]

        # colors = [matplotlib.colors.rgb2hex(x) for x in colors]

        # custom_colors = ["#3399CC", "#FF6666", "#3CB371"]

        # custom_colors = ["#80a591", "#f3c38f", "#bc8ec0"]

        # custom_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        indices = [0, 2, 3]
        custom_colors = [colors[i] for i in indices]
        custom_colors_rgb = [colors_rgb[i] for i in indices]

        print(custom_colors_rgb)

        dims = list(dim_order.values())[:-1]
        for i, dim in enumerate(dims):
            ax.patches[dim].set_facecolor(custom_colors[i])
            ax.patches[dim].set_edgecolor(custom_colors[i])

        # Give a legend to each of these colors (increase, decrease, relevant control, irrelevant control)
        # labels = ["Increase", "Decrease", "Relevant Control"]
        # handles = [
        #     plt.Rectangle((0, 0), 1, 1, color=custom_colors[i]) for i in range(3)
        # ]
        # # ax.legend(
        # #     handles, labels, bbox_to_anchor=(0.5, 1.25), loc="upper center", ncol=2
        # # )
        sns.despine()

        ax.set_xlabel("Dimension")
        ax.set_ylabel("Predicted Difference")

        # xticks = np.arange(10, 70, ).tolist()
        # xticks.insert(0, 1)
        # xticks.insert(-1, 68)

        xticks = [10, 20, 30, 40, 50, 60]

        xticklabels = [str(x) for x in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        # ax.set_xticks(np.arange(0, len(all_diffs), 10))
        # ax.set_xticklabels(np.arange(0, len(all_diffs), 10))

        fig.savefig(
            os.path.join(out_path, img_name.split(".")[0] + "_histogram.pdf"),
            dpi=300,
            bbox_inches="tight",
            transparent=False,
            pad_inches=0.05,
        )
        plt.close(fig)

        # dims = [*dim_order.values()]
        # codes_orig = codes_orig[dims]
        # codes_manip = codes_manip[dims]
        # diff_predictions = codes_manip - codes_orig

        # fig, ax = plt.subplots(figsize=(12, 3))
        # sns.set_context("paper", font_scale=1)
        # sns.set_style("white")

        # custom_colors = ["#3399CC", "#FF6666", "#3CB371", "#DAA520"]
        # sns.set_palette(custom_colors)
        # sns.barplot(
        #     y=diff_predictions,
        #     x=dims,
        #     order=dims,
        #     ax=ax,
        # )

        # sns.despine()

        # ax.grid(False)
        # labels = ["Increase", "Decrease", "Relevant Control", "Irrelevant Control"]
        # # labels = ["Increase", "Decrease", "Relevant Control"]
        # ax.set_xticklabels(labels, fontsize=12)
        # ax.set_ylabel("Difference in Prediction", fontsize=12)
        # plt.tight_layout()
        # plt.show()

        # for ext in ["png", "pdf"]:
        #     fname = os.path.join(f_path, f"{img_base}_histogram.{ext}")
        #     fig.savefig(
        #         fname, bbox_inches="tight", dpi=300, pad_inches=0, transparent=False
        #     )
        # plt.close(fig)

        for (name, index), color in zip(dim_order.items(), custom_colors):
            fig_img = plot_dim_3x2(images, Y, index, top_k=10)
            for ext in ["pdf", "png"]:
                fname = os.path.join(f_path, f"{img_base}_{name}_dim_{index}.{ext}")
                fig_img.savefig(fname, bbox_inches="tight", dpi=300, pad_inches=0)
            plt.close(fig_img)
        orig_img = Image.open(orig_path)
        orig_img.save(os.path.join(f_path, img_name.replace(".jpg", ".pdf")))
        manip_img = Image.open(manip_path)
        manip_img.save(os.path.join(f_path, img_name.replace(".jpg", "_manip.pdf")))


if __name__ == "__main__":
    args = parse_args()
    run_causal_comparison(args.embedding_path, args.img_root)
