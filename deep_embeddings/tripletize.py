#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from deep_embeddings import extract_features
from deep_embeddings import Sampler
from deep_embeddings import ExperimentParser

parser = ExperimentParser(description='Extract features and tripletize from a dataset using a pretrained model and module')
parser.add_argument('--in_path', type=str, default='./data/image_data/things', help='Path to image dataset or any other dataset')
parser.add_argument('--out_path', type=str, default='./data/triplets', help='Path to store image features')
parser.add_argument('--model_name', type=str, default='vgg16_bn', help='Name of the model to use if we want to extrct features')
parser.add_argument('--module_name', type=str, default='classifier.3', help='Name of the module to use')
parser.add_argument('--extract', action="store_true", default=False, help='Extract features from the dataset')
parser.add_argument('--tripletize', action="store_true", default=False, help='Tripletize the features')
parser.add_argument('--adaptive', action="store_true", default=False, help='Adaptively sample triplets')
parser.add_argument('--n_samples', type=int, default=int(2e7), help='Number of samples to use for tripletization')
parser.add_argument('--seed', type=int, default=0, help='Random seed for tripletization')


def run_pipeline(in_path, out_path="", extract=False, tripletize=False, n_samples=2e6, seed=42, adaptive=False, 
                 model_name="vgg16_bn", module_name="classifier.3"):
    
    if extract:
        print("Extracting features from model: {} and module: {}".format(model_name, module_name))
        if not out_path:
            out_path = os.path.join("./data", "triplets", model_name, module_name)
        if not os.path.exists(out_path):
            print(f"Creating output path: {out_path}")
            extract = True
            os.makedirs(out_path)

        extract_features(in_path, out_path, model_name, module_name, batch_size=2)
        in_path = os.path.join(out_path, "features.npy")

    if tripletize:
        n_mio_samples = str(int(n_samples // 1e6)) + "mio"
        out_path = os.path.join(out_path, "triplets_{}".format(n_mio_samples))
        print("Start sampling triplets for the model...")
        sampler = Sampler(in_path, out_path, n_samples=n_samples, k=3, train_fraction=0.9, seed=seed)
        sampler.run_and_save_tripletization(adaptive)
        print("... Done!")


if __name__ == '__main__':
    args = parser.parse_args()
    run_pipeline(args.in_path, 
                 args.out_path,
                 extract=args.extract, 
                 tripletize=args.tripletize, 
                 n_samples=args.n_samples, 
                 seed=args.seed, 
                 adaptive=args.adaptive, 
                 model_name=args.model_name,
                 module_name=args.module_name)
