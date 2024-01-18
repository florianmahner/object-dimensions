"""
Important Question: What do we want to achieve?
e.g. how can we make the representations more human like?
"""

import torch
import glob
import os
import joblib
import pickle
import torch.nn as nn
import torch.nn.functional as F
from imagenet import ImageNetValidationDataset
import torchvision
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset, random_split


# TODO I can also extract the imagenet features beforehand since, they are fixed.


def compute_mse(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    return (y - y_hat).mul(0.5).pow(2).mean()


def compute_accuracy(y: torch.Tensor, probas_hat: torch.Tensor) -> float:
    return (y == torch.argmax(probas_hat, dim=1)).sum() / len(y)


def compute_human_triplet_loss(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    pass


def freeze_parameters_(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def load_linear_weights(path: str) -> nn.Linear:
    files = sorted(glob.glob(os.path.join(path, "*.joblib")))

    n_dim = len(files)
    n_features = joblib.load(files[0]).coef_.shape[0]
    weights = nn.Parameter(torch.zeros(n_dim, n_features), requires_grad=False)
    for i, file in enumerate(files):
        weights[i] = torch.from_numpy(joblib.load(file).coef_)
    weights.requires_grad = True
    return weights


class Extractor:
    """Simple extractor for any model and any module from torchvision. Used here to
    extract imagenet validation features"""

    def __init__(self, dnn_model: nn.Module, module_name: str, device: torch.device):
        self.dnn_model = dnn_model.to(device)
        self.module_name = module_name
        self.activations = None
        self.device = device

        def get_activations():
            def hook(model, input, output):
                self.activations = output.detach()

            return hook

        func = get_activations()
        self.trace_named_modules(dnn_model)
        self.traced_modules[self.module_name].register_forward_hook(func)

    def trace_named_modules(self, model: nn.Module) -> None:
        names, modules = zip(*model.named_modules())
        self.traced_modules = dict(zip(names, modules))

    @torch.no_grad()
    def __call__(self, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        feature_arr = []
        label_arr = []
        for x, y in tqdm(dataloader):
            x = x.to(self.device)
            self.dnn_model(x)
            feature_arr.append(self.activations)
            label_arr.append(y)
            self.activations = None
        feature_arr = torch.cat(feature_arr, dim=0)
        label_arr = torch.cat(label_arr, dim=0)
        save_dict = {"features": feature_arr, "labels": label_arr}
        return save_dict


class InverseClassifier(nn.Module):
    def __init__(
        self,
        dnn_model: nn.Module,
        regression_path: str,
        freeze: bool = True,
    ):
        super().__init__()
        self.dnn_model = dnn_model
        self._init_linear_model(regression_path)

        if freeze:
            freeze_parameters_(self.dnn_model)
            # self.weights.requires_grad = False

        self.clf = self.dnn_model.classifier[-1]

    def _init_linear_model(self, regression_path):
        self.weights = load_linear_weights(regression_path)
        n = len(self.weights)
        self.bias = nn.Parameter(torch.randn(n), requires_grad=True)

    def forward(self, features: torch.Tensor):
        dimensions = F.relu(features @ self.weights.T)
        # imagenet specific dimension bias. lukas makes it different somehow
        dimensions = dimensions * self.bias.abs()
        y_hat = dimensions @ torch.pinverse(self.weights).T
        probas_hat = F.softmax(self.clf(y_hat), dim=1)
        return probas_hat


class Trainer(nn.Module):
    def __init__(self, predictor: nn.Module):
        super().__init__()
        self.predictor = predictor
        self.optim = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)

    def trainstep(self, x: torch.Tensor, y: torch.Tensor):
        self.optim.zero_grad()
        probas_hat = self.predictor(x)
        ce_loss = F.cross_entropy(probas_hat, y)
        ce_loss.backward()
        self.optim.step()
        accuracy = compute_accuracy(y, probas_hat)
        return accuracy

    def valstep(self, x: torch.Tensor, y: torch.Tensor):
        probas_hat = self.predictor(x)
        accuracy = compute_accuracy(y, probas_hat)
        return accuracy


def create_train_val_dataloaders(features, labels, batch_size=128, val_ratio=0.2):
    dataset = TensorDataset(features, labels)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader


def main():
    regression_path = "results/dnn/variational/vgg16_bn/classifier.3/20.mio/sslab/150/256/1.0/14/analyses/sparse_codes"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    dataset = ImageNetValidationDataset(root="./data/ILSVRC")
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, drop_last=False)

    dnn_model = torchvision.models.vgg16_bn(weights="IMAGENET1K_V1")

    extract_features = False
    if extract_features:
        extractor = Extractor(dnn_model, "classifier.3", device)
        out = extractor(dataloader)
        path = "./data/misc/imagenet_features/vgg16"
        features = out["features"]
        labels = out["labels"]
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "features_labels.pkl"), "wb") as f:
            pickle.dump(out, f)

    else:
        path = "./data/misc/imagenet_features/vgg16"
        with open(os.path.join(path, "features_labels.pkl"), "rb") as f:
            out = pickle.load(f)
        features = out["features"]
        labels = out["labels"]

    train_dataloader, val_dataloader = create_train_val_dataloaders(
        features, labels, batch_size=64
    )

    inverse_clf = InverseClassifier(dnn_model, regression_path).to(device)
    trainer = Trainer(inverse_clf)
    trainer = trainer.to(device)

    # Base accuracy
    readout = inverse_clf.clf
    y_base = readout(features).cpu()
    acc_base = compute_accuracy(labels, y_base)
    print(f"Base accuracy: {acc_base:.4f}")

    for epoch in range(100):
        # Training
        trainer.train()  # Set the model to training mode
        train_acc = 0.0
        for features, labels in train_dataloader:
            features, labels = features.to(device), labels.to(device)
            acc = trainer.trainstep(features, labels)
            train_acc += acc
        train_acc /= len(train_dataloader)

        # Validation
        trainer.eval()  # Set the model to evaluation mode
        val_acc = 0.0
        with torch.no_grad():  # Disable gradient computation during validation
            for features, labels in val_dataloader:
                features, labels = features.to(device), labels.to(device)
                acc = trainer.valstep(features, labels)
                val_acc += acc
        val_acc /= len(val_dataloader)

        print(
            f"Epoch: {epoch}, Training Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}"
        )


if __name__ == "__main__":
    main()
