#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from tomlparse import argparse

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


def extract_features(
    img_root: str,
    out_path: str,
    model_name: str = "vgg16_bn",
    module_name: str = "classifer.3",
    batch_size: int = 4,
):
    """Extract features from a dataset using a pretrained model"""
    from thingsvision.utils.storing import save_features
    from thingsvision.utils.data import DataLoader, ImageDataset
    from thingsvision.core.extraction import center_features

    extractor = load_model(model_name)
    out_path = os.path.join(out_path, model_name, module_name)

    dataset = ImageDataset(
        root=img_root,
        out_path=out_path,
        transforms=extractor.get_transformations(),
        backend=extractor.get_backend(),
    )
    assert len(dataset) > 0, "Dataset from path {} is empty!".format(img_root)

    batches = DataLoader(
        dataset=dataset, batch_size=batch_size, backend=extractor.get_backend()
    )

    features = extractor.extract_features(
        batches=batches, module_name=module_name, flatten_acts=False
    )

    if model_name in ["clip", "OpenCLIP"]:
        features = center_features(features)

    save_features(features, out_path, file_format="npy")


def load_model(model_name: str):
    from thingsvision import get_extractor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name in ["clip", "OpenCLIP"]:
        model = get_extractor(
            model_name="OpenCLIP",
            pretrained=True,
            device=device,
            source="custom",
            # I think what we used is the resnet50.visual version for the paper.
            model_parameters={"variant": "RN50", "dataset": "laion2b_s32b_b79k"},
        )
    elif model_name == "barlowtwins-rn50":
        model = get_extractor(
            model_name="barlowtwins-rn50",
            pretrained=True,
            device=device,
            source="ssl",
        )
    elif model_name == "VGG16_ecoset":
        model = get_extractor(
            model_name="VGG16_ecoset",
            pretrained=True,
            device=device,
            source="custom",
        )
    elif model_name == "cornet-s":
        model = get_extractor(
            model_name="cornet-z",
            pretrained=True,
            device=device,
            source="custom",
        )
    elif model_name == "densenet121":
        model = get_extractor(
            model_name="densenet121",
            pretrained=True,
            device=device,
            source="timm",
        )
    else:
        model = get_extractor(
            model_name,
            pretrained=True,
            device=device,
            source="torchvision",
        )

    return model


if __name__ == "__main__":
    args = parse_args()
    extract_features(
        img_root=args.img_root,
        out_path=args.out_path,
        model_name=args.model_name,
        module_name=args.module_name,
        batch_size=args.batch_size,
    )
