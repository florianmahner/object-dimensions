import os
import matplotlib

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from tomlparse import argparse
from objdim.utils import (
    DimensionPredictor,
    load_sparse_codes,
    load_image_data,
    plot_dim_3x2,
)
from pathlib import Path


# Each entry of dims is sorted by increase, decrease, relevant control, irrelevant control
CAUSAL_DIMS = dict()
CAUSAL_DIMS["toilet.jpg"] = {
    "increase": 25,
    "decrease": 10,
    "relevant_control": 13,
    "irrelevant_control": 8,
}
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


def load_img_and_predict_codes(path, predictor):
    img = Image.open(path).convert("RGB")
    codes = predictor(img)[1]
    codes = codes.detach().numpy()
    return codes


def run_causal_comparison(embedding_path, img_root):
    model_name, module_name = Path(embedding_path).parts[-3:-1]

    save_path = os.path.join("./results", "experiments", model_name)
    linear_weights_path = os.path.join(save_path, "linear_weights")

    predictor = DimensionPredictor(model_name, module_name, "cpu", linear_weights_path)
    causal_root = os.path.join("./data", "image_data", "causal")
    out_path = os.path.join(save_path, "causal_comparison")
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
        sns.set_style("ticks")
        ax = sns.barplot(
            x=np.arange(0, len(all_diffs), 1),
            y=all_diffs,
            edgecolor=".8",
            facecolor=(0, 0, 0, 0),
        )
        colors = sns.color_palette("deep")

        # give colors in rgb 0-255
        colors_rgb = [
            (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)) for x in colors
        ]

        indices = [0, 2, 3]
        custom_colors = [colors[i] for i in indices]
        custom_colors_rgb = [colors_rgb[i] for i in indices]

        dims = list(dim_order.values())[:-1]
        for i, dim in enumerate(dims):
            ax.patches[dim].set_facecolor(custom_colors[i])
            ax.patches[dim].set_edgecolor(custom_colors[i])

        sns.despine(offset=10)
        ax.set_xlabel("Dimension Number")
        ax.set_ylabel("Predicted Difference")
        xticks = [10, 20, 30, 40, 50, 60]
        xticklabels = [str(x) for x in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        fig.savefig(
            os.path.join(out_path, img_name.split(".")[0] + "_histogram.pdf"),
            dpi=300,
            bbox_inches="tight",
            transparent=False,
            pad_inches=0.05,
        )
        plt.close(fig)

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
