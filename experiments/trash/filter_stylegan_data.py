import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import os
from PIL import Image
import multiprocessing
import glob
from functools import partial


def compute_features(images, model):
    """Compute the features of all the images using the specified model"""
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    features = []
    for img in images:
        img_tensor = transform(img).unsqueeze(0)
        feature = model.features(img_tensor)
        feature = feature.view(feature.size(0), -1)
        features.append(feature.detach().numpy())
    features = np.concatenate(features, axis=0)
    return features


def compute_similarity(features, method="cosine"):
    """Compute the pairwise similarity between features using the specified method"""
    if method == "cosine":
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        similarity = features @ features.T
    elif method == "euclidean":
        similarity = -np.sqrt(
            ((features[:, np.newaxis, :] - features[np.newaxis, :, :]) ** 2).sum(
                axis=-1
            )
        )
    else:
        raise ValueError("Invalid similarity method")
    return similarity


def compute_features_batch(batch_paths, model):
    """Compute VGG16 features for a batch of images"""
    batch_images = [Image.open(img_path) for img_path in batch_paths]
    batch_features = compute_features(batch_images, model)
    return batch_features


def cluster_images(img_dir, n_clusters=10):
    """Cluster images in a directory based on VGG16 features"""
    model = models.vgg16(pretrained=True).features.eval()
    img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
    features = compute_features([Image.open(img_path) for img_path in img_paths], model)
    clusters = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(features)
    clusters = [
        list(np.array(img_paths)[np.where(clusters == i)]) for i in range(n_clusters)
    ]
    return clusters


if __name__ == "__main__":
    img_dir = "./data/stylegan_dataset_centroids"
    clusters = cluster_images(img_dir, n_clusters=10, batch_size=1000, num_workers=4)
    # create a new
