#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import joblib
import torch
import glob

from thingsvision import get_extractor
import torch.nn as nn
import torch.nn.functional as F


class LatentPredictor(nn.Module):
    """ Predicts Sparse Codes  (i.e. embedding dimensions) from a collection of sampled images.""" 

    def __init__(self, model_name='vgg_16bn', module_name='classifier.3', device='cpu', regression_path=""):
        super().__init__()
        extractor = get_extractor(model_name=model_name, pretrained=True, device=device, source='torchvision')
        model = extractor.model
        model = model.eval()
        self.feature_extractor = model.features
        self.classifier = model.classifier
        self.classifier_trunc = model.classifier[:int(module_name[-1])+1] # we only compute up to the classifying layer that we want to use 
        self.pooling = model.avgpool
        self.transforms = extractor.get_transformations() # Image transformations of the model!

        # Find the number of regression weights, we have one regression weight per dimension.
        self.embedding_dim = len(glob.glob(os.path.join(regression_path, '*.joblib')))

        self.device = device

        n_clf_features = self.classifier_trunc[-1].out_features
        self.regression = nn.Linear(n_clf_features, self.embedding_dim)

        self._update_weights(regression_path)
        
    def _update_weights(self, path):
        W, b = self._load_regression_weights(path)
        self.regression.weight.data = W.to(self.device)
        self.regression.bias.data = b.to(self.device)

    def _load_regression_weights(self, path):
        # this loads the regression weight in the decoding layer!
        in_size = self.regression.in_features
        out_size = self.regression.out_features
        W = torch.zeros(out_size, in_size)
        b = torch.zeros(out_size)
        i = 0
        for f in sorted(os.listdir(path)):
            if f.endswith('joblib'):
                W[i] += joblib.load(os.path.join(path, f)).coef_
                b[i] += joblib.load(os.path.join(path, f)).intercept_
                i += 1
        return W, b

    def forward(self, x):
        """ Forward pass of the model give an image. First extracts VGG feature representations and then predicts the sparse codes from these
        using the learned regression weights. """

        assert x.dtype == torch.uint8, "Image data must be uint8 in range (0,255)"

        # First do the transformations as has been done for the feature extractions
        x = self.transforms(x)

        features = self.feature_extractor(x)
        features = self.pooling(features).reshape(x.shape[0], -1)

        # gives class probabilities on imagenet for that image?
        logits = self.classifier_trunc(features)
        probas = F.softmax(logits, dim=1)
        
        behavior_features = self.classifier_trunc(features)
        latent_codes = self.regression(behavior_features)
        latent_codes = F.relu(latent_codes)
        
        return probas, latent_codes

    @torch.no_grad()
    def predict_codes_from_features(self, features):
        latent_codes = self.regression(features)
        latent_codes = F.relu(latent_codes)
        
        return latent_codes.squeeze()

    @torch.no_grad()
    def extract_features_from_img(self, img):
        features = self.feature_extractor(img)
        features = self.pooling(features).reshape(img.shape[0], -1)
        logits = self.classifier_trunc(features)
        probas = F.softmax(logits, dim=1)

        return logits, probas

    @torch.no_grad()
    def predict_codes_from_img(self, img):
        logits, _ = self.extract_features_from_img(img)
        latent_codes = self.predict_codes_from_features(logits)
        latent_codes = F.relu(latent_codes)

        return latent_codes