import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deep_embeddings.utils.utils import img_to_uint8, load_deepnet_activations, load_sparse_codes, load_image_data
from deep_embeddings import ExperimentParser
from deep_embeddings.utils.latent_predictor import LatentPredictor
from thingsvision import get_extractor
from experiments.visualization.visualize_embedding import plot_dim

from PIL import Image


parser = ExperimentParser(description="Causal comparison")
parser.add_argument("--embedding_path", type=str, default="./results", help="Path to final embedding")
parser.add_argument("--dnn_path", type=str, default="./data/triplets/vgg16_bn/classifier.3", help="Path to DNN activity")
parser.add_argument("--img_root", type=str, default="./data/things_plus", help="Path to image root directory")


FNAMES = ["toilet.jpg", 'bottle.jpg', "basketball.jpg", "fire_hydrant.jpg", "footprint.jpg", "lemon.jpg"]

def load_and_predict_img(path, predictor):
    # Load the image path using PIL and convert to torch tensor
    img = Image.open(path).convert("RGB")
    codes = predictor.predict_codes_from_img(img)    
    codes = codes.detach().numpy()
    return codes


def plot_causal_comparison(embedding_path, dnn_path, img_root):
    base_path = os.path.dirname(os.path.dirname(embedding_path))
    regression_path = os.path.join(base_path, "analyses", "sparse_codes")

    predictor = LatentPredictor(regression_path=regression_path, device="cpu")

    causal_root = os.path.join("./data", "causal_images")
    out_path = os.path.join(base_path, "analyses", "causal_comparison")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    X = load_deepnet_activations(dnn_path, center=True, zscore=False, relu=True, to_torch=True)

    images = load_image_data(img_root)[0]
    Y = load_sparse_codes(embedding_path)
    

    for fname in FNAMES:
        orig_path = os.path.join(causal_root, "original", fname)
        codes_orig = load_and_predict_img(orig_path, predictor)
        manip_path = os.path.join(causal_root, "manipulated", fname)
        codes_manip = load_and_predict_img(manip_path, predictor)
    
        # diff_predictions = np.abs(codes_orig - codes_manip)
        diff_predictions = codes_manip - codes_orig

        # Get only the top 20 difference in predictions and set the rest to 0
        # top_20_indices = np.argsort(-diff_predictions)[:20]

        # # Create a bool array of the top 20 indices of len diff_predictions
        # top_20_bool = np.zeros(len(diff_predictions), dtype=bool)
        # top_20_bool[top_20_indices] = True
        # diff_predictions = diff_predictions * top_20_bool

        fig, ax  = plt.subplots(figsize=(8, 6))
        bar_list = ax.bar(range(len(diff_predictions)), diff_predictions, fill =False, edgecolor='black', linewidth=0.4)

        # Find indices of top 3 largest differences in predicitons
        top_3_top = np.argsort(-diff_predictions)[:3]
        top_3_bottom = np.argsort(diff_predictions)[:3]

        for color, index_top, index_bottom in zip(["red", "green", "blue"], top_3_top, top_3_bottom):
            bar_list[index_top].set_color(color)
            bar_list[index_top].set_fill(True)

            bar_list[index_bottom].set_color(color)
            bar_list[index_bottom].set_fill(True)

            # Make a rotated elongated  arrow pointing to the bar at the index with the index, rotate the arrow 45 degrees
            ax.annotate(index_top, xy=(index_top, diff_predictions[index_top]), xytext=(index_top, diff_predictions[index_top] + 0.1),
                        arrowprops=dict(facecolor=color, shrink=0.05, width=0.5, headwidth=3, headlength=5), rotation=45)
            
            ax.annotate(index_bottom, xy=(index_bottom, diff_predictions[index_bottom]), xytext=(index_bottom, diff_predictions[index_bottom] + 0.1),
                        arrowprops=dict(facecolor=color, shrink=0.05, width=0.5, headwidth=3, headlength=5), rotation=45)
            

            # Plot the dimension of the top 3 largest differences in predictions
            for index, name in zip([index_top, index_bottom], ["increase", "decrease"]):
                fig_img = plot_dim(images, Y, index, top_k=10)
                fig_img.suptitle(f"Dim {index}")
                fig_name = os.path.join(out_path, fname.split(".")[0] + "_{}_dim_{}.png").format(name, index)
                fig_img.savefig(fig_name, dpi=300, bbox_inches="tight", pad_inches=0)
                plt.close(fig_img)


            
        ax.set_xlabel("Latent dimension")
        ax.set_ylabel("Manipulated - Original")
        fig.tight_layout()

    
        fig_name = os.path.join(out_path, fname.split(".")[0] + "_histogram.png")
        fig.savefig(fig_name, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


        # Copy the original image and the manipulated image to the output directory
        orig_img = Image.open(orig_path)
        orig_img = orig_img.resize((224, 224))
        orig_img.save(os.path.join(out_path, fname))
        manip_img = Image.open(manip_path)
        manip_img = manip_img.resize((224, 224))
        manip_img.save(os.path.join(out_path, fname.split(".")[0] + "_manip.png"))


if __name__ == "__main__":
    args = parser.parse_args()
    plot_causal_comparison(args.embedding_path, args.dnn_path, args.img_root)