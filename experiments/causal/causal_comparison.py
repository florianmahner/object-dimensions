
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deep_embeddings.utils.utils import img_to_uint8
from deep_embeddings import ExperimentParser
from deep_embeddings.utils.latent_predictor import LatentPredictor

from PIL import Image

""" 

NOTES

Loading the wrap maximally activates the food dimension. Good sanity check
High dimensions are not very meaningful it seems like. Why?
"""


parser = ExperimentParser(description="Causal comparison")
parser.add_argument("--embedding_path", type=str, default="./restuls", help="Path to final embedding")


FNAMES = ['wrap_18s.jpg', 'bathtub.jpg', "toilet.jpg", "basketball.jpg"]

def load_and_predict_img(path, predictor):
    # Load the image path using PIL and convert to torch tensor
    img = Image.open(path).convert("RGB")
    img = np.array(img)
    img = torch.Tensor(img)
    img = img_to_uint8(img)
    img = img.T    
    img = img.unsqueeze(0)


    
    code = predictor(img)[1].squeeze()
    code = code.detach().numpy()

    return code


def plot_causal_comparison(embedding_path):
    base_path = os.path.dirname(os.path.dirname(embedding_path))
    regression_path = os.path.join(base_path, "analyses", "sparse_codes")

    predictor = LatentPredictor(regression_path=regression_path, device="cpu")

    img_root = os.path.join("./data", "causal_images")
    out_path = os.path.join(base_path, "analyses", "causal_comparison")
    if not os.path.exists(out_path):
        os.makedirs(out_path)


    for fname in FNAMES:
        orig_path = os.path.join(img_root, "original", fname)
        codes_orig = load_and_predict_img(orig_path, predictor)

        breakpoint()


        manip_path = os.path.join(img_root, "manipulated", fname)
        codes_manip = load_and_predict_img(manip_path, predictor)
        

        diff_predictions = np.abs(codes_manip - codes_orig)

        bar_list = plt.bar(range(len(diff_predictions)), diff_predictions, fill =False, edgecolor='black', linewidth=0.4)

        # Find indices of top 3 largest differences in predicitons
        top_3_indices = np.argsort(-diff_predictions)[:3]


        breakpoint()

        for color, index in zip(["red", "green", "blue"], top_3_indices):
            bar_list[index].set_color(color)
            bar_list[index].set_fill(True)

            # Make a rotated elongated  arrow pointing to the bar at the index with the index
            plt.annotate(index, xy=(index, diff_predictions[index]), xytext=(index, diff_predictions[index] + 0.3), arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=2))


        plt.xlabel("Latent dimension")
        plt.ylabel("Absolute difference in prediction")

        
                
        fig_name = os.path.join(out_path, fname.split(".")[0] + "_histogram.png")
        plt.savefig(fig_name, dpi=300, bbox_inches="tight", pad_inches=0)

        break


if __name__ == "__main__":

    args = parser.parse_args()

    args.embedding_path = "./embedding/1/params/parameters.npz"
    plot_causal_comparison(args.embedding_path)