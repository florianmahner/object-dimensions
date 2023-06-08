import os
import glob
import pickle
from PIL import Image
import cv2


import torch
from thingsvision import get_extractor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


""" Complet forward pass on image set
FInd images with that score high in multiple dimensions! """


class Regression:
    def __init__(self, path, device):
        super().__init__()
        self._init_network(path, device)

    def _init_network(self, path, device):
        """Loads a regression predictor from a npy file"""
        try:
            betas = np.load(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file {path}")

        n_features = betas.shape[0]
        n_dim = betas.shape[1]
        self.regression = nn.Linear(n_dim, n_features)
        self.regression.weight.data = torch.from_numpy(betas.T).float().to(device)
        self.regression.bias.data = torch.zeros(n_dim).float().to(device)

    def __call__(self, x):
        x = self.regression(x)
        x = F.relu(x)
        x = x.squeeze()
        return x

    def predict(self, x):
        return self(x)


class ImageDataset:
    def __init__(self, img_root):
        self.img_root = img_root
        self._find_image_paths()

    def _find_image_paths(self):
        """Find all images ending with .jpg in image_root recursively"""
        path = os.path.join(self.img_root, "**", "*.jpg")
        self.filepaths = glob.glob(path, recursive=True)
        self.filepaths = sorted(self.filepaths)

    def get_image(self, fname):
        """Get image from dataset by filename"""
        # FInd fname in samples
        for path in self.filepaths:
            if fname in path:
                break
        img = Image.open(path).convert("RGB")
        fname = os.path.basename(path).split(".")[0]
        return img, fname

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        img = Image.open(path).convert("RGB")
        fname = os.path.basename(path).split(".")[0]
        return img, fname

    def __len__(self):
        return len(self.filepaths)


class DimPred(nn.Module):
    def __init__(self, predictor, scaler, device):
        super().__init__()
        self.mean = torch.tensor(scaler.mean_).float().to(device)
        self.std = torch.tensor(scaler.scale_).float().to(device)

        source = "custom"
        model_name = "OpenCLIP"
        model_parameters = {"variant": "RN50x64", "dataset": "openai"}

        self.extractor = get_extractor(
            model_name=model_name,
            pretrained=True,
            device=device,
            source=source,
            model_parameters=model_parameters,
        )
        self.predictor = predictor
        self.transforms = self.extractor.get_transformations()
        self.device = device

        self.model = self.extractor.model

        self.saliency_layer = getattr(self.model.visual, "layer4")
        self.attn_pool = getattr(self.model.visual, "attnpool")
        layers = list(self.model.visual.children())[
            :-1
        ]  # Remove the last layer (ie. attnblock)
        self.feature_extractor = nn.Sequential(*layers)

    def convert_net_to_matrix(self):
        """Converts the network to a matrix representation"""

    def get_activations(self, img, transform=True):
        if transform:
            img = self.transforms(img)
        if len(img.shape) == 3:
            img = img.view(1, *img.shape)
        img = img.to(self.device)
        features = self.feature_extractor(img)
        return features

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def forward(self, img, transform=True):
        """Forward pass of the model give an image. First extract feature representations and then
        predicts the sparse codes from these using the learned regression weights."""
        features = self.get_activations(img, transform=transform)
        h = features.register_hook(self.activations_hook)
        attn_pool = self.attn_pool(features)

        # Normalize the attention pooled features
        attn_pool = (attn_pool - self.mean) / self.std
        dimensions = self.predictor.predict(attn_pool)

        return dimensions


def reshape_vit_features(tensor, height=14, width=14):
    breakpoint()
    fmap = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    fmap = fmap.permute(0, 3, 1, 2)
    return fmap


def find_gradient_heatmap_(img, regression_predictor, latent_dim=1):
    """Extracts the gradients set by a hook in the regression predicto"""
    img.requires_grad = True

    # Do a forward pass whilte preserving the graph
    with torch.enable_grad():
        dim_predict = regression_predictor(img, transform=True)
        latent = dim_predict[latent_dim]
        latent.backward()
        gradients = regression_predictor.get_activations_gradient()

    # Pool  the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Get the activations of the transformerd vit layer
    activations = regression_predictor.get_activations(img)[0].detach()

    # Idea: the sensitivity of activations to a target class can be
    # understood as the importance of the activation map to the class
    # (given by the gradient), hence we weight the activation maps
    # with the gradients
    for i in range(4096):
        activations[i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=0).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.cpu().numpy()
    return heatmap


def superimpose_heatmaps(img, heatmap, fname, dim):
    img = np.array(img)
    heatmap = (heatmap * 255).astype(np.uint8)

    heatmap = cv2.resize(heatmap, (224, 224))
    img = cv2.resize(img, (224, 224))

    # Invert heatmap so that 255 is 0 and 0 is 255
    heatmap = cv2.bitwise_not(heatmap)
    cmap = cv2.COLORMAP_JET
    heatmap_img = cv2.applyColorMap(heatmap, cmap)
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img)
    axes[1].imshow(super_imposed_img)
    for ax in axes:
        ax.axis("off")

    path = f"./experiments/philipp_causal/heatmaps/{dim}"
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(os.path.join(path, f"{fname}.png"), dpi=300)
    plt.close(fig)


def run_causal():
    img_root = "./data/image_data/reference_images"
    regression_path = "./experiments/philipp_causal/coefs.npy"
    scaler_path = "./experiments/philipp_causal/Xstandardizer_49d_elastic_OpenCLIP-RN50x64-openai_visual.pkl"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    scaler = pickle.load(open(scaler_path, "rb"))

    img_loader = ImageDataset(img_root)
    regression_net = Regression(regression_path, device)
    dimension_predictor = DimPred(regression_net, scaler, device)

    save_predictions = False
    if save_predictions:
        predictions = np.zeros((len(img_loader), 49))

        for i, (img, fname) in enumerate(img_loader):
            print(f"Processing image {i}", end="\r")
            dimvals = dimension_predictor(img, transform=True)

            breakpoint()
            dimvals = dimvals.detach().cpu().numpy()
            predictions[i] = dimvals

        with open("./experiments/philipp_causal/predictions.npy", "wb") as f:
            np.save(f, predictions)

    predictions = np.load("./experiments/philipp_causal/predictions.npy")

    filepaths = img_loader.filepaths

    topk_indices = np.argsort(-predictions, axis=0)
    topk = 8

    # for dim in range(49):
    for dim in [1]:
        print(f"Dimension {dim}")
        topk_per_dim = topk_indices[:topk, dim]
        fpaths = [filepaths[i] for i in topk_per_dim]

        # for f in fpaths:
        #     img, fname = img_loader.get_image(f)
        #     heatmap = find_gradient_heatmap_(img, dimension_predictor, latent_dim=dim)
        #     superimpose_heatmaps(img, heatmap, fname, dim)

        fname = "apple.jpg"
        img, fname = img_loader.get_image(fname)

        # for img, fname in img_loader:
        heatmap = find_gradient_heatmap_(img, dimension_predictor, latent_dim=dim)
        superimpose_heatmaps(img, heatmap, fname, dim)


if __name__ == "__main__":
    run_causal()
