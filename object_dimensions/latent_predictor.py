#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import joblib
import torch
import glob

from thingsvision import get_extractor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import torch.nn as nn
import torchvision.transforms as T


def load_regression_weights(path, in_features, out_features, to_numpy=False):
    # This loads the regression weight in the decoding layer!
    W = torch.zeros(out_features, in_features, dtype=torch.float32)
    b = torch.zeros(out_features, dtype=torch.float32)
    # Dims are sorted when saving, so joblib_00 is the dimension with largest magnitude and so on.
    files = glob.glob(os.path.join(path, "*.joblib"), recursive=True)
    # Sort the fileendings to get the correct order of dimensions.
    files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    for i, f in enumerate(files):
        W[i] += joblib.load(f).coef_
        b[i] += joblib.load(f).intercept_

    if to_numpy:
        W = W.numpy()
        b = b.numpy()

    return W, b


class LatentPredictor(nn.Module):
    """Predicts Sparse Codes  (i.e. embedding dimensions) from a collection of sampled images."""

    def __init__(
        self,
        model_name="vgg16_bn",
        module_name="classifier.3",
        device="cpu",
        regression_path="",
    ):
        super().__init__()
        if isinstance(device, torch.device):
            device_type = device.type
            self.device = device
        else:
            device_type = device
            self.device = torch.device(device)
            
        extractor = get_extractor(
            model_name=model_name, pretrained=True, device=device_type, source="torchvision"
        )
        model = extractor.model
        model = model.eval()

        self.feature_extractor = model.features

        # Get up the second to last element of the featrue extractor
        self.feature_extractor = nn.Sequential(
            *list(self.feature_extractor.children())[:-1]
        )

        # Get the maxpooling layer
        self.maxpool = model.features[-1]

        self.classifier = model.classifier
        self.classifier_trunc = model.classifier[
            : int(module_name[-1]) + 1
        ]  # we only compute up to the classifying layer that we want to use
        self.pooling = model.avgpool
        self.transforms = (
            extractor.get_transformations()
        )  # Image transformations of the model!

        # Find the number of regression weights, we have one regression weight per dimension.
        self.embedding_dim = len(glob.glob(os.path.join(regression_path, "*.joblib")))
        assert self.embedding_dim > 0, "No regression weights found!"

        self.device = device
        n_clf_features = self.classifier_trunc[-1].out_features
        self.regression = nn.Linear(n_clf_features, self.embedding_dim)
        self._update_weights(regression_path)

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def _update_weights(self, path):
        W, b = load_regression_weights(
            path, self.regression.in_features, self.regression.out_features
        )
        self.regression.weight.data = W.to(self.device)
        self.regression.bias.data = b.to(self.device)

    def forward(self, x, transform=True):
        """Forward pass of the model give an image. First extracts VGG feature representations and then
        predicts the sparse codes from these using the learned regression weights."""

        # First do the transformations as has been done for the feature extraction, if not done in the dataloader
        probas, features = self.extract_features_from_img(x, transform=transform)

        # NOTE Negative valus are also discared when building the triplets and for the sparse code predictions
        latent_codes = self.predict_codes_from_features(features)

        return probas, latent_codes

    def predict_codes_from_features(self, features):
        features = F.relu(features)
        latent_codes = self.regression(features)
        latent_codes = F.relu(latent_codes)
        return latent_codes.squeeze()

    def get_activations(self, img, transform=True):
        """Extracts the activations of the last layer of the feature extractor prior
        to pooling and classification etc."""
        if transform:
            img = self.transforms(img)
        if img.dim() == 3:
            img = img.view(1, *img.shape)
        img = img.to(self.device)
        features = self.feature_extractor(img)

        return features, img

    def extract_features_from_img(self, img, transform=True):
        features, img = self.get_activations(img, transform=transform)
        # We register the hook before the maxpool layer of the feature extractor,
        # so that the resulting heatmap is twice the size
        h = features.register_hook(self.activations_hook)
        self.maxpool(features)

        features = self.pooling(features).reshape(img.shape[0], -1)
        logits = self.classifier_trunc(features)
        features = F.relu(logits)
        probas = F.softmax(logits, dim=1)
        return probas, features

    def predict_codes_from_img(self, img, transform=True):
        _, latent_codes = self.forward(img, transform=transform)
        return latent_codes
