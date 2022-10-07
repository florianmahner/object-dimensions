#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle

import pandas as pd
import numpy as np
from collections import defaultdict

from deep_embeddings.utils.utils import load_sparse_codes


def remove_rare_features(features, descriptions, freq):
    reduced_indices = np.where(features.sum(1) > freq)[0]
    features = features.iloc[reduced_indices, :]
    # descriptions = [descriptions[r] for r in reduced_indices]
    descriptions = descriptions[reduced_indices]
    return features, descriptions

def cosine(x, y):
    return (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))

def pearsonr(x, y):
    x_c = x - x.mean()
    y_c = y - y.mean()
    return cosine(x_c, y_c)

def correlate_features(w, features):
    return [pearsonr(w, feature) for feature in features]

def label_dimensions(dimensions, features, descriptions, top_k):
    dim_labels = defaultdict(list)
    features = features.to_numpy() 
    for i, w_i in enumerate(dimensions.T):
        sorted_objects = np.argsort(-w_i)
        features = features[:, sorted_objects]
        w_i = w_i[sorted_objects]
        corrs = correlate_features(w_i, features)
        topk_features = np.argsort(corrs)[::-1][:top_k]
        dim_labels[i].extend(descriptions[topk_features].tolist())
    return dim_labels

def save_dim_labels_(out_path, dim_labels):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with open(os.path.join(out_path, 'dimensions_labelled.txt'), 'wb') as f:
        f.write(pickle.dumps(dim_labels))
    dim_labelled = pd.DataFrame.from_dict(dict(dim_labels))
    dim_labelled.to_csv(os.path.join(out_path, 'dimensions_labelled.csv'), sep=',', index=False)

# load feature norm matrix
feature_norms = pd.read_csv('./feature_object_matrix.csv')
feature_norms.rename(columns={ feature_norms.columns[0]: "features" }, inplace = True)
descriptions = feature_norms['features'].to_numpy()
feature_norms = feature_norms.drop(columns=['features'])

item_names = pd.read_csv('./item_names.tsv', sep='\t', encoding='utf-8')

path_dnn = '/LOCAL/fmahner/DeepEmbeddings/learned_embeddings/weights_vgg_12_512bs/params/pruned_q_mu_epoch_500.txt'
act_path = '/LOCAL/fmahner/THINGS/vgg_bn_features_12'

w_dnn = load_sparse_codes(path_dnn, with_dim=False)
filenames = np.loadtxt(os.path.join(act_path, 'file_names.txt'), dtype=str)

ref_indices = np.array([i for i, f in enumerate(filenames) if "b.jpg" in f])

w_dnn = w_dnn[ref_indices, :]

feature_norms, descriptions = remove_rare_features(feature_norms,  descriptions, 30)
dim_labels = label_dimensions(w_dnn, feature_norms, descriptions, 15)
save_dim_labels_('./gpt3_labels', dim_labels)

