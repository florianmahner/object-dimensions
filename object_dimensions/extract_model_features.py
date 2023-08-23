#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch

from thingsvision.utils.storing import save_features
from thingsvision import get_extractor
from thingsvision.utils.data import DataLoader
from thingsvision.core.extraction import center_features

from object_dimensions.image_dataset import ImageDataset
from tomlparse import argparse

sys.path.append("./stylegan_xl")
from stylegan_xl import legacy
from stylegan_xl.dnnlib import util

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract features from a dataset using a pretrained model"
    )
    parser.add_argument(
        "--img_root", type=str, default="./data/THINGS", help="Path to image dataset"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./data/vgg_features",
        help="Path to save features",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for the dataloader to extract features",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vgg16_bn",
        help="Name of the pretrained model to use",
    )
    parser.add_argument(
        "--module_name",
        type=str,
        default="classifier.3",
        help="Name of the layer to extract features from",
    )
    return parser.parse_args()


def load_model(model_name):
    if model_name in ["clip", "OpenCLIP"]:
        model = get_extractor(
            model_name="OpenCLIP",
            pretrained=True,
            device=device,
            source="custom",
            model_parameters={"variant": "ViT-H-14", "dataset": "laion2b_s32b_b79k"},
        )
    else:
        model = get_extractor(
            model_name, pretrained=True, device=device, source="torchvision"
        )

    return model


def extract_features(img_root, out_path, model_name, module_name, batch_size):
    """Extract features from a dataset using a pretrained model"""
    extractor = load_model(model_name)
    dataset = ImageDataset(
        root=img_root,
        out_path=out_path,
        transforms=extractor.get_transformations(),
        backend=extractor.get_backend(),
    )
    assert len(dataset) > 0, "Dataset from path {} is empty!".format(img_root)

    filenames = dataset.images
    with open(out_path + "/filenames.txt", "w") as f:
        f.write("\n".join(filenames))

    batches = DataLoader(
        dataset=dataset, batch_size=batch_size, backend=extractor.get_backend()
    )

    features = extractor.extract_features(
        batches=batches, module_name=module_name, flatten_acts=True
    )

    if model_name in ["clip", "OpenCLIP"]:
        features = center_features(features)
    save_features(features, out_path, file_format="npy")


if __name__ == "__main__":
    args = parse_args()
    extract_features(
        img_root=args.img_root,
        out_path=args.out_path,
        model_name=args.model_name,
        module_name=args.module_name,
        batch_size=args.batch_size,
    )
