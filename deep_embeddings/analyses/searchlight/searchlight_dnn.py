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

from dataset import ImageDataset
from collections import defaultdict
from os.path import exists as pexists
from os.path import join as pjoin
from sklearn.linear_model import ElasticNet
from typing import Tuple, List, Any, Dict, Iterator

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--dataset_name', type=str,
        choices=['things', 'imagenet'],
        help='name of dataset')
    aa('--in_path', type=str,
        help='path to image dataset')
    aa('--activation_path', type=str,
        help='path to DNN features for each image in the dataset')
    aa('--regression_path', type=str,
        help='path to ElasticNet models (that were trained to predicted SPoSE dimensions given DNN features)')
    aa('--model_name', type=str, default='vgg16_bn',
        choices=['alexnet', 'resnet50', 'resnet101', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'])
    aa('--module', type=str, default='classifier.3')
    aa('--model_path', type=str, default=None,
        help='directory where to load torchvision model weights from')
    aa('--window_size', type=int, default=20,
        choices=[15, 20, 25, 30, 35],
        help='size of window to be masked in image')
    aa('--stride', type=int, default=3,
        choices=[1, 2, 3, 4, 5])
    aa('--analysis', type=str, default='regression',
        choices=['classification', 'regression', 'feature_extraction'])
    aa('--latent_dim', type=int,
        help='perform searchlight for top k objects within respective *latent dimension*')
    aa('--top_k', type=int, default=10,
        help='perform searchlight for the top k *activated* images within the union (combined set) of THINGS and ImageNet')
    aa('--device', type=str, default='cpu',
        choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'])
    aa('--rnd_seed', type=int, default=42)
    args = parser.parse_args()
    return args

def load_features(activation_path:str) -> np.ndarray:
    with open(pjoin(activation_path, 'features.npy'), 'rb') as f:
        features = np.load(f)
    return features

def get_file_names(activation_path:str) -> List[str]:
    return open(pjoin(activation_path, 'file_names.txt'), 'r').read().splitlines()

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
                #mask a window of size ws x ws centered around position (i, j)
                masked_img = mask_img(img_copy, i, j, window_size)
                out = model(masked_img)
                out = out.squeeze(0)
                probas = F.softmax(out, dim=0)
                top_k_preds = torch.argsort(probas, descending=True).cpu().numpy()[:top_k]
                masked_predictions[(i, j)] = {idx2cls[pred]: float(probas[pred]) for pred in top_k_preds}
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
                     dataset_name:str,
                     in_path:str,
                     activation_path:str,
                     regression_path:str,
                     model_name:str,
                     module_name:str,
                     window_size:int,
                     stride:int,
                     analysis:str,
                     latent_dim:int,
                     top_k:int,
                     device:torch.device,
) -> None:
    #load THINGS objects and ImageNet categories into memory and find common classes between the two datasets
    item_names, sortindex = utils.load_inds_and_item_names()
    imagenet_classes = utils.parse_imagenet_classes('./data/imagenet1000_classes.txt')
    idx2cls = dict(enumerate(imagenet_classes))
    cls2idx = {cls:idx for idx, cls in idx2cls.items()}

    transforms = utils.compose_transforms()
    dataset = ImageDataset(
                            root=in_path,
                            out_path='',
                            things=True if dataset_name == 'things' else False,
                            things_behavior=False,
                            add_ref_imgs=True if dataset_name == 'things' else False,
                            transforms=transforms,
                            )

    predictor = joblib.load(pjoin(regression_path, f'predictor_{latent_dim:02d}.joblib'))
    file_names = get_file_names(activation_path)
    features = load_features(activation_path)
    dimension = predictor.predict(features)
    top_k_samples = np.argsort(-dimension)[:top_k]

    global activations
    activations = {}
    model = register_hook(model)
    model.to(device)
    model.eval()

    out_path = pjoin('searchlights', 'dnn', dataset_name, analysis, 'top_k', f'{latent_dim:02d}')
    if not os.path.exists(out_path):
        print('\n...Creating directories.\n')
        os.makedirs(out_path)

    #store top 5 predictions for each position in masked image
    predictions = {}
    for i, k in enumerate(top_k_samples):
        img = dataset[k][0][None, ...]
        img = img.to(device)
        out = model(img)
        y_pred_orig = dimension[k]
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
        predictions[file_names[k]] = masked_predictions
        with open(pjoin(out_path, f'searchlight_ws_{window_size:02d}_stride_{stride:02d}_{i:02d}.npy'), 'wb') as f:
            np.save(f, deltas)

    utils.pickle_file(predictions, out_path, 'masked_predictions')

if __name__ == '__main__':
    args = parseargs()
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)

    model, _ = vision.load_model(args.model_name, pretrained=False if args.model_path else True, device=args.device, model_path=args.model_path)

    search_img_space(
                    model=model,
                    dataset_name=args.dataset_name,
                    in_path=args.in_path,
                    activation_path=args.activation_path,
                    regression_path=args.regression_path,
                    model_name=args.model_name,
                    module_name=args.module,
                    window_size=args.window_size,
                    stride=args.stride,
                    analysis=args.analysis,
                    latent_dim=args.latent_dim,
                    top_k=args.top_k,
                    device=args.device,
                    )
