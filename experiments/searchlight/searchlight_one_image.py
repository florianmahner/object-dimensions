#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import torch
import os
import pickle


import numpy as np
import matplotlib.pyplot as plt
from thingsvision import get_extractor
from thingsvision.utils.data import ImageDataset

from object-dimensions import ExperimentParser

from object-dimensions.utils.searchlight_utils import mask_img
from object-dimensions.utils.utils import load_deepnet_activations, img_to_uint8
from object-dimensions.utils.latent_predictor import LatentPredictor

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
    "--feature_path",
    type=str,
    default="./data/models/vgg16bn/classifier.3",
    help="Path to the stored features.",
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
# QUERIES =  [
#     ('flashlight_01b', [32, 43]),
#     ('ball_01b', [13, 34]),
#     ('wine_01b', [12, 20])
# ]


# QUERIES =  [
#     ('ball_01b', [34, 54])
# ]

QUERIES = [("toilet_plus", [18])]


def searchlight_(img, regression_predictor, window_size, stride=1, latent_dim=1):
    H, W = img.shape[-2:]
    diffs = torch.zeros((H, W))

    # NOTE we could also index previous feature mat but dont need to do this for now!
    # TODO Check if I really only want to do this for one dimension at a time?
    dim_original = regression_predictor.predict_codes_from_img(img)

    dim_original = dim_original[latent_dim]

    with torch.no_grad():
        for i in np.arange(0, H, stride):
            print(
                f"\n...Currently masking image centered around row position {i}.",
                end="\r",
            )
            for j in np.arange(0, W, stride):
                img_copy = img.clone()

                # mask a window of size ws x ws centered around position (i, j)
                masked_img = mask_img(img_copy, i, j, window_size)
                dim_masked = regression_predictor.predict_codes_from_img(masked_img)
                dim_masked = dim_masked[latent_dim]

                p_explained = torch.abs(dim_original - dim_masked) / (
                    dim_original + 1e-12
                )  # NOTE add this to avoid division by zero
                diffs[i : i + stride, j : j + stride] = p_explained

    diffs = diffs.cpu().numpy()

    return diffs


def search_image_spaces(
    base_path,
    regression_predictor,
    dataset,
    img_idx,
    window_size=20,
    stride=1,
    latent_dims=[1, 2, 3],
    device="cpu",
):
    out_path = os.path.join(
        base_path, "analyses", "searchlights", "dnn", "img_{}".format(img_idx)
    )
    if not os.path.exists(out_path):
        print("\n...Creating directories.\n")
        os.makedirs(out_path)

    for dim in latent_dims:
        print(
            f"\n...Currently performing searchlight analysis for latent dimension {dim}.\n"
        )
        img = dataset[img_idx].to(device).unsqueeze(0)

        # breakpoint()
        # img = img.squeeze().mT.cpu().numpy()
        # img = img_to_uint8(img)

        diffs = searchlight_(img, regression_predictor, window_size, stride, dim)
        img = img.squeeze().mT.cpu().numpy()
        save_dict = {"diffs": diffs, "img": img, "dim": dim}

        # store save dict as pickle file in directory
        with open(
            os.path.join(
                out_path,
                f"./searchlight_ws_{window_size}_stride_{stride}_dim_{dim}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(save_dict, f)

        # save diffs as image fille
        fig, ax = plt.subplots(1, 1)
        ax.imshow(diffs, cmap="RdBu_r")
        fig.tight_layout()
        fig.savefig(
            os.path.join(out_path, f"./searchlight_diffs_dim_{dim}.png"), dpi=300
        )

    fig, ax = plt.subplots(1, 1)
    img = img_to_uint8(img)
    ax.imshow(img.T)
    fig.savefig(os.path.join(out_path, f"./searchlight_img.png"), dpi=300)


def map_predictions_to_classes(predictions, classes):
    """ Map predictions to classes """
    return [classes[pred] for pred in predictions]


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    extractor = get_extractor(
        model_name=args.model_name, pretrained=True, device=device, source="torchvision"
    )
    dataset = ImageDataset(
        root=args.img_root,
        out_path="./",
        backend=extractor.get_backend(),
        transforms=extractor.get_transformations(),
    )

    base_path = os.path.dirname(os.path.dirname(args.embedding_path))
    regression_path = os.path.join(base_path, "analyses", "sparse_codes")

    predictor = LatentPredictor(
        args.model_name, args.module_name, device, regression_path
    )
    predictor.to(device)

    features = load_deepnet_activations(args.feature_path, to_torch=True)
    features = features.to(device)

    idx2obj = dataset.idx_to_cls
    obj2idx = dataset.cls_to_idx

    # Find the images in the dataset that correspond to the query!
    for query, latent_dims in QUERIES:
        img_idx = [i for i, img in enumerate(dataset.images) if query in img][0]
        search_image_spaces(
            base_path,
            predictor,
            dataset,
            img_idx,
            args.window_size,
            args.stride,
            latent_dims,
            device,
        )

