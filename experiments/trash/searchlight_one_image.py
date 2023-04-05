#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import torch
import os
import pickle
import cv2


import numpy as np
import matplotlib.pyplot as plt
from thingsvision import get_extractor
from PIL import Image, ImageDraw
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
from object_dimensions import ExperimentParser

from experiments.visualization.visualize_embedding import plot_dim

from object_dimensions.utils.utils import img_to_uint8, load_image_data
from object_dimensions.utils.latent_predictor import LatentPredictor

parser = ExperimentParser(description="Searchlight analysis for one image.")
parser.add_argument("--embedding_path", type=str, help="Path to the embedding file.")
parser.add_argument(
    "--img_root",
    type=str,
    default="./data/images",
    help="Path to the all images used for the embedding.",
)
parser.add_argument(
    "--analysis", type=str, default="regression", help="Type of analysis to perform."
)
parser.add_argument(
    "--model_name", type=str, default="vgg16_bn", help="Name of the model to use."
)
parser.add_argument(
    "--module_name", type=str, default="classifier.3", help="Name of the module to use."
)
parser.add_argument(
    "--window_size",
    type=int,
    default=20,
    choices=[15, 20, 25, 30, 35],
    help="Size of the window to use for the searchlight.",
)
parser.add_argument(
    "--stride",
    type=int,
    default=1,
    choices=[1, 2, 3, 4, 5],
    help="Stride of the window to use for the searchlight.",
)
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed to use for the searchlight."
)

# The queries of the database that we want to visualize!

QUERIES = [
    ("flashlight_01b", [58, 101, 7, 120, 37,9,69 ])
    # ("basketball_plus", [82, 63, 64, 40, 23, 52, 68])
    # ("wineglass_plus", [4, 69, 66])
    # ("wine_01b", [9, 4, 66, 67,36, 69, 4, 47])
]

def searchlight_(img, regression_predictor, window_size, stride=1, latent_dim=1, device="cpu"):    
    reshape = T.Compose(
        [T.Resize(256), T.CenterCrop(224), T.ToTensor()]
    )
    normalize = T.Compose(
        [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    img_reshaped = reshape(img)
    img_normed = normalize(img_reshaped)

    dim_original = regression_predictor(img_normed, transform=False)[1]
    dim_original = dim_original[latent_dim]

    # Attach grad to img_normed
    img_normed.requires_grad = True

    # Do a forward pass and preserve gradients
    with torch.enable_grad():
        dim_predict = regression_predictor(img_normed, transform=False)[1]
        latent = dim_predict[latent_dim]

        # Do a backward pass
        latent.backward()

        # Get the gradients of the image
        gradients = regression_predictor.get_activations_gradient()


    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = regression_predictor.get_activations(img)[0].detach()

    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()

    heatmap = F.relu(heatmap)

    # normalize the heatmap
    heatmap /= torch.max(heatmap)
    return heatmap.cpu().numpy()

def search_image_spaces(
    base_path,
    regression_predictor,
    dataset,
    img_idx,
    image_name,
    window_size=20,
    stride=1,
    latent_dims=[1, 2, 3],
    device="cpu",
):
    out_path = os.path.join(
        base_path, "analyses", "searchlights", "{}".format(image_name)
    )
    if not os.path.exists(out_path):
        print("\n...Creating directories.\n")
        os.makedirs(out_path)

    img = Image.open(dataset[img_idx])
    # codes = regression_predictor(img, transform=True)[1]
    # codes = codes.detach().cpu().numpy()
    reshape = T.Compose(
        [T.Resize(256), T.CenterCrop(224)]
    )
    img_vis = reshape(img)
    img_vis = np.array(img_vis)
    fig, ax = plt.subplots(1, 1)
    
    ax.imshow(img_vis)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(
        os.path.join(out_path, f"img.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )

    for dim in latent_dims:
        print(
            f"\n...Currently performing searchlight analysis for latent dimension {dim}."
        )

        diffs = searchlight_(img, regression_predictor, window_size, stride, dim, device)
        save_dict = {"diffs": diffs, "img": img_vis, "dim": dim}

        # store save dict as pickle file in directory
        with open(os.path.join(out_path, f"./searchlight_dim_{dim}.pkl"), "wb") as f:
            pickle.dump(save_dict, f)
        
        heatmap = diffs    
        heatmap = img_to_uint8(heatmap)
        heatmap = cv2.resize(heatmap, (img_vis.shape[0], img_vis.shape[1]))

        # Invert heatmap so that 255 is 0 and 0 is 255
        heatmap = cv2.bitwise_not(heatmap)
        # heatmap = cv2.GaussianBlur(heatmap, (13, 13), sigmaX=11, sigmaY=11)
        cmap = cv2.COLORMAP_JET
        heatmap_img = cv2.applyColorMap(heatmap, cmap)
        super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img_vis, 0.5, 0)
        path = os.path.join(out_path, f"./searchlight_dim_{dim}_superimposed.png")
        fig = plt.figure()
        plt.imshow(super_imposed_img)
        plt.axis("off")
        plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)

        fig, ax = plt.subplots(1, 1)
        ax.imshow(diffs, cmap="RdBu_r")
        # Turn off all the ticks
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(
            os.path.join(out_path, f"./searchlight_dim_{dim}.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        # fig = plot_dim(dataset, codes, dim, 10)
        # fig.savefig(
        #     os.path.join(out_path, f"{dim}_topk.png"),
        #     dpi=300,
        #     bbox_inches="tight",
        #     pad_inches=0
        # )


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = get_extractor(
        model_name=args.model_name, pretrained=True, device=device, source="torchvision"
    )
    images, indices = load_image_data(args.img_root)

    base_path = os.path.dirname(os.path.dirname(args.embedding_path))
    regression_path = os.path.join(base_path, "analyses", "sparse_codes")

    predictor = LatentPredictor(
        args.model_name, args.module_name, device, regression_path
    )
    predictor.to(device)

    # Find the images in the dataset that correspond to the query!
    for query, latent_dims in QUERIES:
        img_idx = [i for i, img in enumerate(images) if query in img][0]
        search_image_spaces(
            base_path,
            predictor,
            images,
            img_idx,
            query,
            args.window_size,
            args.stride,
            latent_dims,
            device,
        )
