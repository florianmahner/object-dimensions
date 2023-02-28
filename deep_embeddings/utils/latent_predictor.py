#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import joblib
import torch
import glob

from thingsvision import get_extractor
import torch.nn as nn
import torch.nn.functional as F


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
        self.extractor = get_extractor(
            model_name=model_name, pretrained=True, device=device, source="torchvision"
        )
        self.module_name = module_name
        for m in self.extractor.model.named_modules():
            if m[0] == module_name:
                self.n_features = m[1].in_features
                break

        # Find the number of regression weights, we have one regression weight per dimension.
        self.embedding_dim = len(glob.glob(os.path.join(regression_path, "*.joblib")))
        self.regression = nn.Linear(self.n_features, self.embedding_dim)

        self.transforms = (
            self.extractor.get_transformations()
        )  # Image transformations of the model!
        self.device = device
        self._update_weights(regression_path)

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

    @torch.no_grad()
    def predict_codes_from_features(self, features):
        features = F.relu(features)
        latent_codes = self.regression(features)
        latent_codes = F.relu(latent_codes)
        return latent_codes.squeeze()

    @torch.no_grad()
    def extract_features_from_img(self, img, transform=True):
        if transform:
            img = self.transforms(img)
            img = img.view(1, 1, *img.shape) if img.dim() == 3 else img

        features = self.extractor.extract_features(
            img, self.module_name, flatten_acts=True, output_type="tensor"
        )

        probas = F.softmax(features, dim=1)
        features = F.relu(features)  # use non linearity after softmax

        return probas, features

    @torch.no_grad()
    def predict_codes_from_img(self, img):
        _, latent_codes = self.forward(img)
        return latent_codes
