#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

from deep_embeddings import extract_features
from deep_embeddings import Sampler

parser = argparse.ArgumentParser(description='Extract features and tripletize from a dataset using a pretrained model and module')
parser.add_argument('--img_root', type=str, default='./data/THINGS', help='Path to image dataset')
parser.add_argument('--extract', type=bool, default=True, help='Extract features from the dataset')
parser.add_argument('--tripletize', type=bool, default=True, help='Tripletize the features')
parser.add_argument('--adaptive', type=bool, default=False, help='Adaptively sample triplets')
parser.add_argument('--n_samples', type=int, default=1e6, help='Number of samples to use for tripletization')
parser.add_argument('--rnd_seed', type=int, default=42, help='Random seed for tripletization')


def run_pipeline(img_root, extract=False, tripletize=False, n_samples=2e6, rnd_seed=42, adaptive=False):
    # I want to take: 1 supervised, 1 unsupervised, e.g. the latents of a GAN and one transformer based!
    # model_names = ('vgg_16bn', 'stylegan_xl', 'OpenCLIP')
    # module_names = ('classifier.3', 'latent', 'visual')

    model_names = ['alexnet', 'clip']
    module_names = ['classifier.3', 'visual']

    for model_name, module_name in zip(model_names, module_names):
        print("Extracting features from model: {} and module: {}".format(model_name, module_name))

        out_path = os.path.join("./data", "models", model_name, module_name)
        if not os.path.exists(out_path):
            print(f"Creating output path: {out_path}")
            extract = True
            os.makedirs(out_path)

        if extract:
            extract_features(img_root, out_path, model_name, module_name, batch_size=1)

    
        feature_path = os.path.join(out_path, "features.npy")
        n_mio_samples = str(int(n_samples // 1e6)) + "mio"
        out_path = os.path.join(out_path, "triplets_{}".format(n_mio_samples))

        if tripletize:
            print("Start sampling triplets for the model...")
            sampler = Sampler(feature_path, out_path, n_samples=n_samples, k=3, train_fraction=0.9, rnd_seed=42)
            sampler.run_and_save_tripletization(adaptive)


if __name__ == '__main__':
    args = parser.parse_args()
    # behavior_path = "./data/reference_images"
    # dataset_path = "./data/image_data/images12"
    args.img_root = "./data/image_data/images12"
    run_pipeline(args.img_root, 
                 extract=args.extract, 
                 tripletize=args.tripletize, 
                 n_samples=args.n_samples, 
                 rnd_seed=args.rnd_seed, 
                 adaptive=args.adaptive)







    

