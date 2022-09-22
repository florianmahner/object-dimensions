#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import torch
import numpy as np
from thingsvision.core import Extractor
from thingsvision.utils.data import ImageDataset


from searchlight_utils import mask_img
import sys
sys.path.append('../image_generation')
from latent_predictor import LatentPredictor


class Config:
    latent_path = "./"
    regression_path = "./"
    dnn_path = "./"

    model_name = "vgg16_bn"
    module = "./classifier.3"
    latent_version = 'sampled'

    image_root = './LOCAL/fmahner/THINGS/image_data/images12'

    analysis  = 'regression' # or classification / feature extraction
    latent_dim = 0 # do for specific dim
    n_codes = 100 # number of embedding dims, extract manually!

    truncation = 0.4
    topk_k = 16
    device = 'cuda:0'
    rdn_seed = 42



def searchlight_(img, model, regression_predictor, window_size, stride=1):
    H, W = img.shape[-2:]
    diffs = np.zeros((H, W))

    features_img = model(img)
    dim_original = regression_predictor.predict(features_img)

    with torch.no_grad():
        for i in np.arange(0, H, stride):
            print(f'\n...Currently masking image centered around row position {i}.')
            for j in np.arange(0, W, stride):
            
                img_copy = img.clone()
                #mask a window of size ws x ws centered around position (i, j)
                masked_img = mask_img(img_copy, i, j, window_size)
                
                features_masked = model(masked_img) # up to the features that we want to extract
                features_masked = features_masked.cpu().numpy()
                dim_masked = regression_predictor.predict(features_masked)

                p_explained = np.abs(dim_original - dim_masked) / dim_original
                diffs[i:i+stride, j:j+stride] = p_explained

    return diffs


def find_topk_images(dim, topk):
    pass


def search_image_spaces(model, regression_predictor, dataset):

    for img in dataset:
        img = img.unsqueeze(0)
        diffs = searchlight_(img, model, regression_predictor)
        print(diffs.shape)
        break



    
    



            
if __name__ == '__main__':

    cfg = Config

    np.random.seed(cfg.rnd_seed)
    random.seed(cfg.rnd_seed)
    torch.manual_seed(cfg.rnd_seed)

    model = Extractor(cfg.model_name, pretrained=False if cfg.dnn_path else True, device=cfg.device, model_path=cfg.dnn_path)
    dataset = ImageDataset(root=cfg.image_root, out_path='./', backend=model.backend, transforms=model.get_transformations())

    device = torch.device(cfg.device)
    predictor = LatentPredictor(cfg.model_name, cfg.module_name, device, cfg.regression_path)

    search_image_spaces(model, predictor, dataset)




