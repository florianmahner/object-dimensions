#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import torchvision

import sys
sys.path.append("./deep_embeddings/analyses/image_generation/stylegan_xl")


from thingsvision.utils.storing import save_features
from thingsvision import Extractor
# from thingsvision.utils.data import ImageDataset, DataLoader
from thingsvision.utils.data import DataLoader
from thingsvision.core.extraction import center_features


from deep_embeddings.utils.image_dataset import ImageDataset

from deep_embeddings.analyses.image_generation.stylegan_xl import legacy
from deep_embeddings.analyses.image_generation.stylegan_xl.dnnlib import util


parser = argparse.ArgumentParser(description='Extract features from a dataset using a pretrained model')
parser.add_argument('--img_root', type=str, default='./data/THINGS', help='Path to image dataset')
parser.add_argument('--out_path', type=str, default='./data/vgg_features', help='Path to save features')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for the dataloader to extract features')
parser.add_argument('--model_name', type=str, default='vgg16_bn', help='Name of the pretrained model to use')
parser.add_argument('--module_name', type=str, default='classifier.3', help='Name of the layer to extract features from')

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=256),
    torchvision.transforms.CenterCrop(size=(224, 224)),
    torchvision.transforms.ToTensor()])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_name):
    # TODO StyleGAN XL not yet working, since discriminator projection to latent is obscure!
    print("Loading model: {} ...".format(model_name))
    if model_name == 'stylegan_xl':
        network_pkl = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet256.pkl"
        print('Loading StyleGAN networks from "%s"...' % network_pkl)
        with util.open_url(network_pkl) as f:
            # We take the discriminator as our feature extractor
            discriminator = legacy.load_network_pkl(f)['D']
            model = discriminator.eval().requires_grad_(False)

    elif model_name == 'clip':
        model =  Extractor(model_name='OpenCLIP', pretrained=True, device=device, source='custom',
                        model_parameters={'variant': 'ViT-H-14', 'dataset': 'laion2b_s32b_b79k'})

    else:
        model =  Extractor(model_name, pretrained=True, device=device, source='torchvision')
    return model


def extract_features(img_root, out_path, model_name, module_name, batch_size):
    model = load_model(model_name)
    dataset = ImageDataset(
            img_root=img_root,
            out_path=out_path,
            transforms=model.get_transformations()
    )

    batches = DataLoader(dataset=dataset, batch_size=batch_size, backend=model.backend)
    features = model.extract_features(
                    batches=batches,
                    module_name=module_name,
                    flatten_acts=True,
                    clip=True if model_name == 'clip' else False,
    )

    if model_name == "clip":
        features = center_features(features) 

    save_features(features, out_path, file_format='npy')


if __name__ == '__main__':
    args = parser.parse_args()
    extract_features(
        img_root=args.img_root,
        out_path=args.out_path,
        model_name=args.model_name,
        module_name=args.module_name,
        batch_size=args.batch_size,
    )