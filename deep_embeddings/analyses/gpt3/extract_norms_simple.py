#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" This script generate GPT3 features for all images from the THINGS image dataset where behavioral data
has been colleted! For each of these images GPT3 features have been generated. This script correlates 
these features with the learned embedding and find the descriptions that best describe each dimension """

import os
import pickle
import argparse

import pandas as pd
import numpy as np
from collections import defaultdict

from deep_embeddings.utils.utils import load_sparse_codes, filter_embedding_by_behavior

parser = argparse.ArgumentParser(description='Label dimensions of sparse codes using GPT3')
parser.add_argument('--embedding_path', default='./embedding.txt', type=str, help='Path to the embedding txt file')
parser.add_argument('--dnn_path', default='./vgg_features', type=str, help='Path to all vgg features')
parser.add_argument('--feature_norm_path', default='./data/feature_norms', type=str, help='Path to GPT3 norms')


def label_dimensions(dimensions, features, descriptions, top_k):
    dim_labels = defaultdict(list)
    features = features.to_numpy() 
    for i, w_i in enumerate(dimensions.T):

        breakpoint()
        sorted_objects = np.argsort(-w_i)
        features = features[:, sorted_objects]
        w_i = w_i[sorted_objects]

        corrs = correlate_features(w_i, features)
        topk_features = np.argsort(corrs)[::-1][:top_k]
        topk_descriptions = descriptions[topk_features].tolist()
        dim_labels[i].extend(topk_descriptions)

    return dim_labels


def save_dim_labels_(out_path, dim_labels):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Write dict to pickle file
    with open(os.path.join(out_path, 'dimensions_labelled.pkl'), 'wb') as f:
        pickle.dump(dim_labels, f)

    dim_labelled = pd.DataFrame.from_dict(dict(dim_labels))
    dim_labelled.to_csv(os.path.join(out_path, 'dimensions_labelled.csv'), sep=',', index=False)

def generate_gpt3_norms(embedding_path, dnn_path, feature_norm_path='./data/feature_norms/feature_object_matrix.csv'):
    feature_norms = pd.read_csv(feature_norm_path)
    feature_norms.rename(columns={ feature_norms.columns[0]: "features" }, inplace = True)
    descriptions = feature_norms['features'].to_numpy()
    feature_norms = feature_norms.drop(columns=['features'])
    # item_names = pd.read_csv('./data/feature_norms/item_names.tsv', sep='\t', encoding='utf-8')

    embedding = load_sparse_codes(embedding_path)
    file_path = os.path.join(dnn_path, 'file_names.txt')

    if not os.path.isfile(file_path):
        raise ValueError('VGG path does not contain file_names.txt')
    else:
        filenames = np.loadtxt(file_path, dtype=str)
    
    # Find reference images in THINGS database, i.e. indices from behavior, i.e. images ending with ".b"
    embedding, filenames = filter_embedding_by_behavior(embedding, filenames)

    dim_labels = label_dimensions(embedding, feature_norms, descriptions, 15)

    base_path = os.path.dirname(os.path.dirname(embedding_path))
    out_path = os.path.join(base_path, 'analyses', 'gpt3_labels')

    save_dim_labels_(out_path, dim_labels)



if __name__ == '__main__':
    args = parser.parse_args()
    generate_gpt3_norms(args.embedding_path, args.dnn_path, args.feature_norm_path)