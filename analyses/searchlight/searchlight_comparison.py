#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import joblib
import os
import random
import re
import time
import torch
import utils

import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision.models as models
import thingsvision.vision as vision

from collections import defaultdict
from os.path import exists as pexists
from os.path import join as pjoin
from sklearn.linear_model import ElasticNet
from typing import Tuple, List, Any, Dict, Iterator

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--dnn_path', type=str,
        help='path to SPoSE DNN embedding matrix')
    aa('--mind_path', type=str,
        help='path to SPoSE behavior embedding matrix')
    aa('--activation_path', type=str, default='./activations/vgg16_bn/classifier.3/THINGS/reduced/',
        help='path to DNN features for each image in the THINGS database (reduced dataset)')
    aa('--regression_path', type=str,
        help='path to ElasticNet models (that were trained to predicted SPoSE dimensions given DNN features)')
    aa('--model_path', type=str, default=None,
        help='directory where to load torchvision model weights from')
    aa('--model_name', type=str, default='vgg16_bn',
        choices=['alexnet', 'resnet50', 'resnet101', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'])
    aa('--module', type=str, default='classifier.3')
    aa('--window_size', type=int, default=20,
        choices=[15, 20, 25, 30, 35],
        help='size of window to be masked in image')
    aa('--stride', type=int, default=3,
        choices=[1, 2, 3, 4, 5])
    aa('--analysis', type=str, default='regression',
        choices=['classification', 'regression'])
    aa('--top_k', type=int, default=8,
        help='perform searchlight for the top k activated and k most dissimilar images across DNNs and human behavior')
    aa('--rho_dim', type=int, default=None,
        help='perform searchlight for top k objects in SPoSE VGG 16 wrt *dim* sorted according to Pearson correlation')
    aa('--duplicates', action='store_true',
        help='whether to perform searchlight for mind-machine comparison with VGG 16 dimensions that allow for duplicates or are unique')
    aa('--device', type=str, default='cpu',
        choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'])
    aa('--rnd_seed', type=int, default=42)
    args = parser.parse_args()
    return args

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def register_hook(model):
    for name, module in model.named_modules():
        module.register_forward_hook(get_activation(name))
    return model

def get_x_window(H:int, y_pos:int, window_size:int) -> Tuple[int]:
    x_top_diff = abs(0 - y_pos)
    mask_x_start = y_pos - x_top_diff if (x_top_diff < window_size) else y_pos - window_size
    x_bottom_diff = abs(H - y_pos)
    mask_x_end = (y_pos + x_bottom_diff + 1) if (x_bottom_diff < window_size) else (y_pos + window_size + 1)
    return mask_x_start, mask_x_end

def get_y_window(W:int, x_pos:int, window_size:int) -> Tuple[int]:
    y_top_diff = abs(0 - x_pos)
    mask_y_start = x_pos - y_top_diff if (y_top_diff < window_size) else x_pos - window_size
    y_bottom_diff = abs(W - x_pos)
    mask_y_end = (x_pos + y_bottom_diff + 1) if (y_bottom_diff < window_size) else x_pos + window_size + 1
    return mask_y_start, mask_y_end

def mask_img(img:torch.Tensor, y_pos:int, x_pos:int, window_size:int) -> torch.Tensor:
    _, _, H, W = img.shape #img shape = [B x C x H x W]
    mask_x_start, mask_x_end = get_x_window(H, y_pos, window_size)
    mask_y_start, mask_y_end = get_y_window(W, x_pos, window_size)
    img[:, :, mask_x_start:mask_x_end, mask_y_start:mask_y_end] = 0 #.5 = grey mask; 0. = black mask
    return img

def searchlight_(
                 analysis:str,
                 model:Any,
                 predictor:Any,
                 img:torch.Tensor,
                 stride:int,
                 window_size:int,
                 idx2cls:dict,
                 module_name:str,
                 y_pred_orig:np.ndarray,
                 top_k:int=5,
                 correct_class=None,
                 ) -> Tuple[np.ndarray, Dict[tuple, dict]]:
    _, _, H, W = img.shape
    deltas = np.zeros((H, W))
    masked_predictions = {}
    with torch.no_grad():
        for i in np.arange(0, H, stride):
            print(f'\n...Currently masking image centered around row position {i}.')
            for j in np.arange(0, W, stride):
                img_copy = img.clone()
                masked_img = mask_img(img_copy, i, j, window_size)
                out = model(masked_img)
                out = out.squeeze(0)
                probas = F.softmax(out, dim=0)
                top_k_preds = torch.argsort(probas, descending=True).cpu().numpy()[:top_k]
                masked_predictions[(i, j)] = {idx2cls[pred]: probas[pred] for pred in top_k_preds}
                if analysis == 'regression':
                    act_masked = activations[module_name]
                    act_masked = act_masked.cpu().numpy()
                    y_pred_masked = predictor.predict(act_masked)
                    p_explained = np.abs(y_pred_orig - y_pred_masked) / y_pred_orig
                    delta = p_explained
                else:
                    p_masked = probas[correct_class]
                    delta = 1 - p_masked
                deltas[i:i+stride, j:j+stride] = delta
    return deltas, masked_predictions

def search_img_space(
                     model,
                     dnn_path:str,
                     mind_path:str,
                     activation_path:str,
                     regression_path:str,
                     model_name:str,
                     module_name:str,
                     window_size:int,
                     stride:int,
                     analysis:str,
                     duplicates:bool,
                     rho_dim:int,
                     top_k:int,
                     device:torch.device,
) -> None:
    #load THINGS objects and ImageNet categories into memory and find common classes between the two datasets
    item_names, sortindex = utils.load_inds_and_item_names()
    imagenet_classes = utils.parse_imagenet_classes('./data/imagenet1000_classes.txt')
    idx2cls = dict(enumerate(imagenet_classes))
    cls2idx = {cls:idx for idx, cls in idx2cls.items()}

    common_classes = utils.get_class_intersection(imagenet_classes, list(item_names))
    #load weight matrix of SPoSE model for DNN activations, and solely extract reference images
    W_dnn, _ = utils.load_sparse_codes(dnn_path)
    transforms = utils.compose_transforms(behavior=True)

    #load references images into memory
    ref_images = utils.load_ref_images(item_names=item_names, compress=False)
    #load targets into memory and find rows that correspond to reference images (to compare dimensions against dimensions found in behavior)
    targets = utils.load_targets(activation_path)
    ref_indices = utils.get_ref_indices(targets)
    W_dnn = W_dnn[ref_indices, :]
    W_dnn = W_dnn[:, np.argsort(-np.linalg.norm(W_dnn, ord=1, axis=0))]
    #load weight matrix of SPoSE model for behavioral data
    #W_b, _ = utils.load_sparse_codes(mind_path)
    W_b = np.loadtxt(mind_path)
    W_b = W_b[sortindex]

    img_indices, img_subset, latent_dim = utils.get_img_subset(W_b, W_dnn, ref_images, rho_dim, top_k, duplicates)
    obj_subset = item_names[img_indices]

    global activations
    activations = {}
    model = register_hook(model)
    model.to(device)
    model.eval()

    #store top 5 predictions for ImageNet classes
    predictions = defaultdict(dict)
    #store image indices that correspond to classes present in ImageNet
    img_identifiers = defaultdict(list)

    out_path = pjoin('searchlights', 'comparison', analysis)
    out_path = pjoin(out_path, 'duplicates', f'{rho_dim:02d}') if duplicates else pjoin(out_path, 'no_duplicates', f'{rho_dim:02d}')
    if not pexists(out_path):
        print('\n...Creating directories.\n')
        os.makedirs(out_path)

    for k, img in enumerate(img_subset):
        obj = obj_subset[k]
        w = W_dnn[img_indices[k]]
        img = transforms(img)[None, ...]
        img = img.to(device)
        out = model(img)
        act_orig = activations[module_name].cpu()

        predictor = joblib.load(pjoin(regression_path, f'predictor_{latent_dim:02d}.joblib'))
        y_pred_orig = predictor.predict(act_orig)
        y_true = w[latent_dim]

        print(f'\nTrue weight value in dimension {latent_dim+1:02d} for reference image no. {img_indices[k]}: {y_true:.3f}\n')
        print(f'\nPredicted weight value in dimension {latent_dim+1:02d} for reference image no. {img_indices[k]}: {y_pred_orig[0]:.3f}\n')

        deltas, masked_predictions = searchlight_(
                                                 analysis=analysis,
                                                 model=model,
                                                 predictor=predictor,
                                                 img=img,
                                                 stride=stride,
                                                 window_size=window_size,
                                                 idx2cls=idx2cls,
                                                 module_name=module_name,
                                                 y_pred_orig=y_pred_orig,
                                                 )

        if k >= top_k:
            subset = 'most_dissim'
            subdir = pjoin(out_path, subset)
            suffix = f'{int(k-top_k):02d}'
        else:
            subset = 'top_k'
            subdir = pjoin(out_path, subset)
            suffix = f'{k:02d}'

        predictions[subset][suffix] = masked_predictions

        if obj in common_classes:
            img_identifiers[subset].append(int(img_indices[k]))
        else:
            img_identifiers[subset].append(0)

        if not os.path.exists(subdir):
            print('\n...Creating directories.\n')
            os.makedirs(subdir)

        with open(pjoin(subdir, f'searchlight_ws_{window_size:02d}_stride_{stride:02d}_{suffix}.npy'), 'wb') as f:
            np.save(f, deltas)

    utils.pickle_file(predictions, out_path, 'masked_predictions')
    utils.pickle_file(img_identifiers, out_path, 'img_identifiers')

if __name__ == '__main__':
    args = parseargs()
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)

    #load pretrained torchvision model into memory
    model, _ = vision.load_model(args.model_name, pretrained=False if args.model_path else True, device=args.device, model_path=args.model_path)

    search_img_space(
                    model=model,
                    dnn_path=args.dnn_path,
                    mind_path=args.mind_path,
                    activation_path=args.activation_path,
                    regression_path=args.regression_path,
                    model_name=args.model_name,
                    module_name=args.module,
                    window_size=args.window_size,
                    stride=args.stride,
                    analysis=args.analysis,
                    duplicates=args.duplicates,
                    rho_dim=args.rho_dim,
                    top_k=args.top_k,
                    device=args.device,
                    )
