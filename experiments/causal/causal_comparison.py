import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deep_embeddings.utils.utils import (
    load_sparse_codes,
    load_image_data,
)
from deep_embeddings import ExperimentParser
from deep_embeddings.utils.latent_predictor import LatentPredictor
from experiments.visualization.visualize_embedding import plot_dim

from PIL import Image

parser = ExperimentParser(description="Causal comparison")
parser.add_argument(
    "--embedding_path", type=str, default="./results", help="Path to final embedding"
)
parser.add_argument(
    "--dnn_path",
    type=str,
    default="./data/triplets/vgg16_bn/classifier.3",
    help="Path to DNN activity",
)
parser.add_argument(
    "--img_root",
    type=str,
    default="./data/things_plus",
    help="Path to image root directory",
)


# Each entry of dims is sorted by increase, decrease, relevant control, irrelevant control
CAUSAL_DIMS = dict()
CAUSAL_DIMS["toilet.jpg"] = {"increase": 37, 
                             "decrease": 12, 
                             "relevant_control": 14, 
                             "irrelevant_control": 70}

CAUSAL_DIMS["bottle.jpg"] = {"increase": 58, 
                             "decrease": 66, 
                             "relevant_control": 47, 
                             "irrelevant_control": 71}

CAUSAL_DIMS["basketball.jpg"] = {"increase": 2, 
                                 "decrease": 100, 
                                 "relevant_control": 52, 
                                 "irrelevant_control": 78}

CAUSAL_DIMS["basketball.jpg"] = {"increase": 86, 
                                 "decrease": 171, 
                                 "relevant_control": 52, 
                                 "irrelevant_control": 49}

def load_and_predict_img(path, predictor):
    # Load the image path using PIL and convert to torch tensor
    img = Image.open(path).convert("RGB")
    codes = predictor.predict_codes_from_img(img)
    codes = codes.detach().numpy()
    return codes


def run_causal_comparison(embedding_path, dnn_path, img_root):
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
        print("Running for ", img_name)
        orig_path = os.path.join(causal_root, "original", img_name)
        codes_orig = load_and_predict_img(orig_path, predictor)
        manip_path = os.path.join(causal_root, "manipulated", img_name)
        codes_manip = load_and_predict_img(manip_path, predictor)

        # Print the dims where the original image activates a lot
        print("Dims important for original image: ", np.argsort(-codes_orig)[:5])
        print("Dims important for manipulated image: ", np.argsort(-codes_manip)[:5])
        print("Dims with largest decrease", np.argsort(codes_manip - codes_orig)[:5])
        print("Dims with largest increase", np.argsort(-codes_manip + codes_orig)[:5])

        dims = [*dim_order.values()]
        codes_orig = codes_orig[dims]
        codes_manip = codes_manip[dims]
        diff_predictions = codes_manip - codes_orig

        fig = plt.figure(figsize=(10, 4))
        sns.set_context("paper", font_scale=1.0)
        sns.set_style("whitegrid")

        # Create a barplot and dont sort the values on the x-axis, but take the order from the dims
        ax = sns.barplot(x=dims, y=diff_predictions, order=dims)

        # Set the labels to [Increase, Decrease, Relevant Control, Irrelevant Control]
        labels = ["Increase", "Decrease", "Control A", "Control B"]
        ax.set_xticklabels(labels)
        ax.set_ylabel("Difference in prediction")

        img_base = img_name.split(".")[0]
        f_path = os.path.join(out_path, img_base) 
        if not os.path.exists(f_path):
            os.makedirs(f_path)

        for name, index in dim_order.items():      
            fig_img = plot_dim(images, Y, index, top_k=10)
            fig_name = os.path.join(f_path, img_base + "_{}_dim_{}.png").format(name, index)
            fig_img.savefig(fig_name, dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close(fig_img)

        for ext in ["pdf", "png"]:
            fname = os.path.join(f_path, f"{img_base}_histogram.{ext}")
            fig.savefig(fname, bbox_inches="tight", dpi=300)

    
        orig_img = Image.open(orig_path)
        orig_img.save(os.path.join(f_path, img_name.replace(".jpg", ".pdf")))
        manip_img = Image.open(manip_path)
        manip_img.save(os.path.join(f_path, img_name.replace(".jpg", "_manip.pdf")))
        


if __name__ == "__main__":
    args = parser.parse_args()
    run_causal_comparison(args.embedding_path, args.dnn_path, args.img_root)
