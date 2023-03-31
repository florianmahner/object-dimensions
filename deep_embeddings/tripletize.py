#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from deep_embeddings import extract_features
from deep_embeddings import Sampler
from deep_embeddings import ExperimentParser

parser = ExperimentParser(description='Extract features and tripletize from a dataset using a pretrained model and module')
parser.add_argument('--img_root', type=str, default='./data/THINGS', help='Path to image dataset')
parser.add_argument('--out_path', type=str, default='', help='Path to store image features')
parser.add_argument('--extract', action="store_true", default=False, help='Extract features from the dataset')
parser.add_argument('--tripletize', action="store_true", default=False, help='Tripletize the features')
parser.add_argument('--adaptive', action="store_true", default=False, help='Adaptively sample triplets')
parser.add_argument('--n_samples', type=int, default=int(2e7), help='Number of samples to use for tripletization')
parser.add_argument('--similarity', type=str, default='dot', choices=['dot', 'cosine'], help='Similarity measure to use for tripletization')
parser.add_argument('--seed', type=int, default=0, help='Random seed for tripletization')


def run_pipeline(img_root, out_path="", extract=False, tripletize=False, n_samples=2e6, seed=42, adaptive=False, similarity="dot"):
    model_names = ["vgg16_bn"]
    module_names = ['classifier.3']

    for model_name, module_name in zip(model_names, module_names):
        print("Extracting features from model: {} and module: {}".format(model_name, module_name))
        if not out_path:
            out_path = os.path.join("./data", "triplets", model_name, module_name)
        if not os.path.exists(out_path):
            print(f"Creating output path: {out_path}")
            extract = True
            os.makedirs(out_path)

        if extract:
            extract_features(img_root, out_path, model_name, module_name, batch_size=2)
    
        feature_path = os.path.join(out_path, "features.npy")
        n_mio_samples = str(int(n_samples // 1e6)) + "mio"
        out_path = os.path.join(out_path, "triplets_{}".format(n_mio_samples))

        if tripletize:
            print("Start sampling triplets for the model...")
            sampler = Sampler(feature_path, out_path, n_samples=n_samples, k=3, 
                              train_fraction=0.9, seed=seed, similarity=similarity)
            sampler.run_and_save_tripletization(adaptive)
            print("... Done!")


if __name__ == '__main__':
    args = parser.parse_args()
    run_pipeline(args.img_root, 
                 args.out_path,
                 extract=args.extract, 
                 tripletize=args.tripletize, 
                 n_samples=args.n_samples, 
                 seed=args.seed, 
                 adaptive=args.adaptive, 
                 similarity=args.similarity)
