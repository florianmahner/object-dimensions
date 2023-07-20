#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["optimize_latents"]

import torchvision
import sys


import random
import torch
import os
import glob

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch.nn as nn
import numpy as np

sys.path.append("./stylegan_xl")
from stylegan_xl import legacy
from stylegan_xl.dnnlib import util
from stylegan_xl.torch_utils import gen_utils
from stylegan_xl.gen_images import make_transform

from object_dimensions.utils.utils import img_to_uint8
from object_dimensions.utils.latent_predictor import LatentPredictor
from torch.utils.data import DataLoader
from torchvision import transforms
from tomlparse import argparse
from PIL import Image


transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize and sample StyleGAN")
    parser.add_argument(
        "--window_size",
        type=int,
        default=50,
        help="Window size of trainer to check convergence of latent dimenisonality",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        default="./weights/params/pruned_q_mu_epoch_300.txt",
        help="Path to weights directory",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vgg16_bn",
        help="Model to load from THINGSvision",
    )
    parser.add_argument(
        "--module_name",
        type=str,
        default="classifier.3",
        help="Layer of the model to load from THINGSvision",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2_000_000,
        help="Number of latent samples to generate",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for sampling"
    )
    parser.add_argument(
        "--truncation",
        type=float,
        default=1.0,
        help="Truncation value for noise sample",
    )
    parser.add_argument(
        "--top_k", type=int, default=16, help="Top k values to sample from"
    )

    parser.add_argument(
        "--sample_dataset",
        default=False,
        action="store_true",
        help="Sample entire dataset",
    )
    parser.add_argument(
        "--find_topk",
        default=False,
        action="store_true",
        help="Find top k latent dimensions",
    )
    parser.add_argument(
        "--optimize_topk",
        default=False,
        action="store_true",
        help="Optimize top k latent dimensions",
    )

    parser.add_argument(
        "--max_iter", type=int, default=200, help="Number of optimizing iterations"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dim", type=int, default=(1, 2), nargs="+", help="Dimension to optimize for"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Weight of the absolute value in a dimension to optimize for",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Weight for the softmax loss in the dimension optimization",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    return parser.parse_args()


def load_style_gan():
    # Load the generator from style gan xl
    network_pkl = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet256.pkl"
    print('Loading networks from "%s"...' % network_pkl)

    with util.open_url(network_pkl) as f:
        generator = legacy.load_network_pkl(f)["G_ema"]
        generator = generator.eval().requires_grad_(False)

    return generator


def find_topk_latents(
    model_name,
    module_name,
    embedding_path,
    n_samples,
    batch_size,
    truncation,
    top_k,
    device,
):
    device = torch.device(device)
    base_path = os.path.dirname(os.path.dirname(embedding_path))
    regression_path = os.path.join(base_path, "analyses", "sparse_codes")
    out_path = os.path.join(base_path, "analyses", "per_dim")

    if not os.path.exists(regression_path):
        raise FileNotFoundError(
            f"Regression path {regression_path} does not exist. Run sparse code predictions \
                                              first before optimizing latents."
        )

    latent_predictor = LatentPredictor(model_name, module_name, device, regression_path)
    predictor = SparseCodesPredictor(
        n_samples=n_samples,
        n_dims=latent_predictor.embedding_dim,
        batch_size=batch_size,
        truncation=truncation,
        top_k=top_k,
        out_path=out_path,
        device=device,
    )
    predictor.predict_latent(latent_predictor)


def optimize_latents(
    model_name,
    module_name,
    embedding_path,
    dim,
    lr,
    max_iter,
    truncation,
    alpha,
    beta,
    window_size,
    device,
):
    base_path = os.path.dirname(os.path.dirname(embedding_path))
    regression_path = os.path.join(base_path, "analyses", "sparse_codes")
    latent_path = os.path.join(base_path, "analyses", "per_dim")

    device = torch.device(device)
    predictor = LatentPredictor(model_name, module_name, device, regression_path)

    generator = load_style_gan()

    trainer = Optimizer(
        lr=lr,
        max_iter=max_iter,
        dim=dim,
        in_path=latent_path,
        truncation=truncation,
        device=device,
        alpha=alpha,
        beta=beta,
        window_size=window_size,
    )

    trainer = Optimizer(
        lr=lr,
        max_iter=max_iter,
        dim=dim,
        in_path=latent_path,
        truncation=truncation,
        device=device,
        alpha=alpha,
        beta=beta,
        window_size=window_size,
    )

    trainer.optimize_latents(generator, predictor)


class StyleGanGenerator(object):
    """Generator class for StyleGan that create a latent and image dataset pair and saves to disk"""

    def __init__(self, out_path, n_samples, truncation, batch_size, device):
        self.n_samples = n_samples
        self.out_path = out_path
        self.truncation = truncation
        self.device = device
        self.batch_size = batch_size

        self.generator = load_style_gan()
        self.generator = self.generator.to(self.device)

    def get_w_from_seed(
        self,
        G,
        batch_sz,
        device,
        truncation_psi=1.0,
        seed=None,
        centroids_path=None,
        class_idx=None,
    ):
        """Get the dlatent from a list of random seeds, using the truncation trick (this could be optional)
        Florian: Adapted this from the original repository to add noise on the centroids!
        """
        dim = G.c_dim
        # dim = 0
        if dim != 0:
            # sample random labels if no class idx is given
            if class_idx is None:
                class_indices = np.random.RandomState(seed).randint(
                    low=0, high=G.c_dim, size=(batch_sz)
                )
                class_indices = torch.from_numpy(class_indices).to(device)
                w_avg = G.mapping.w_avg.index_select(0, class_indices)
            else:
                w_avg = G.mapping.w_avg[class_idx].unsqueeze(0).repeat(batch_sz, 1)
                class_indices = torch.full((batch_sz,), class_idx).to(device)

            labels = F.one_hot(class_indices, G.c_dim)

        else:
            w_avg = G.mapping.w_avg.unsqueeze(0)
            labels = None
            if class_idx is not None:
                print(
                    "Warning: --class is ignored when running an unconditional network"
                )

        z = np.random.RandomState(seed).randn(batch_sz, G.z_dim)
        z = torch.from_numpy(z).to(device)
        w = G.mapping(z, labels)

        # multimodal truncation
        if centroids_path is not None:
            with util.open_url(centroids_path, verbose=False) as f:
                w_centroids = np.load(f)
            w_centroids = torch.from_numpy(w_centroids).to(device)
            w_centroids = w_centroids[None].repeat(batch_sz, 1, 1)

            # add noise to the centroids (added by me)
            noise = torch.randn_like(w_centroids) * 0.01
            w_centroids = w_centroids + noise

            # measure distances
            dist = torch.norm(w_centroids - w[:, :1], dim=2, p=2)
            w_avg = w_centroids[0].index_select(0, dist.argmin(1))

        w_avg = w_avg.unsqueeze(1).repeat(1, G.mapping.num_ws, 1)
        w = w_avg + (w - w_avg) * truncation_psi

        return w

    def sample_latents(self):
        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        translate = (0, 0)
        translate = (0, 0)
        rotate = 0
        if hasattr(self.generator.synthesis, "input"):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            self.generator.synthesis.input.transform.copy_(torch.from_numpy(m))

        sampled_latents = torch.zeros((self.n_samples, 32, 512))
        # seeds = np.random.randint(0, (2**32)-1, self.n_samples)
        seeds = np.arange(self.n_samples)
        centroids_path = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet_centroids.npy"

        for i in range(self.n_samples):
            print(
                f"Generating latent w_space vector: {i+1}/{self.n_samples} ...",
                end="\r",
            )
            w_latent = self.get_w_from_seed(
                self.generator,
                1,
                self.device,
                self.truncation,
                seed=seeds[i],
                centroids_path=centroids_path,
                class_idx=None,
            )

            sampled_latents[i] = w_latent

        return sampled_latents

    def sample_dataset(self, sampled_latents=None):
        """Saves all images and latents to disc"""
        if not os.path.exists(self.out_path):
            print(f"Creating directory {self.out_path}")
            os.makedirs(os.path.join(self.out_path, "images"))
            os.makedirs(os.path.join(self.out_path, "latents"))

        if not sampled_latents:
            sampled_latents = self.sample_latents()

        dataloader = DataLoader(
            sampled_latents, batch_size=self.batch_size, num_workers=4
        )
        print("Generating images and sparse code predictions ...\n")

        n_iter = len(dataloader)
        n_samples = 0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                print("Process Batch {}/{}".format(i + 1, n_iter), end="\r")
                batch = batch.to(self.device)
                print("Process Batch {}/{}".format(i + 1, n_iter), end="\r")
                batch = batch.to(self.device)
                images = gen_utils.w_to_img(self.generator, batch, to_np=True)
                for latent, img in zip(batch, images):
                    img = Image.fromarray(img, mode="RGB")
                    img.save(os.path.join(self.out_path, "images", f"{n_samples}.jpg"))
                    latent = latent.cpu().numpy()
                    np.savez_compressed(
                        os.path.join(self.out_path, "latents", f"{n_samples}.npz"),
                        latent,
                    )
                    n_samples += 1


class StyleGanDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path="./data/stylegan_dataset",
        transform=None,
        return_latent=True,
        n_samples=None,
    ):
        self.latent_paths = sorted(
            glob.glob(os.path.join(base_path, "latents", "*.npz"))
        )
        self.image_paths = sorted(glob.glob(os.path.join(base_path, "images", "*.jpg")))
        random.seed(0)
        random.shuffle(self.latent_paths)
        random.seed(0)
        random.shuffle(self.image_paths)
        self.n_samples = n_samples if n_samples else len(self.latent_paths)
        self.transform = transform
        self.return_latent = return_latent

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.return_latent:
            latent = np.load(self.latent_paths[idx])
            return image, latent

        return image


class SparseCodesPredictor(object):
    """Generate n latent samples a priori for optimization of latent embeddings
    We generate n images for this using the pretrained style gan and then select the topk images that maximally
    activate each of the embedding dimension. We then optimize for these topk latents!
    """

    def __init__(
        self, n_samples, n_dims, batch_size, truncation, top_k, out_path, device
    ):
        self.n_samples = n_samples
        self.n_dims = n_dims
        self.batch_size = batch_size
        self.top_k = top_k
        self.truncation = truncation
        self.out_path = out_path
        self.device = device
        self.dataset = StyleGanDataset(
            base_path="./data/stylegan_dataset_centroids",
            transform=transforms,
            n_samples=n_samples,
        )
        if not os.path.exists(self.out_path):
            print("Creating directories...\n")
            print("Creating directories...\n")
            os.makedirs(self.out_path)

    def predict_latent(self, comparator):
        """Sample latent using StyleGAN Xl batch wise and generate sparse code predictions for all images.
        Store the top k latents and images for each dimension to disk."""
        comparator.to(self.device)

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=8,
        )
        sampled_codes = torch.zeros((len(self.dataset), self.n_dims))
        n_iter = len(dataloader)
        for i, (img, _) in enumerate(dataloader):
            img = img.to(self.device)
            _, codes = comparator(img, transform=False)
            sampled_codes[
                i * self.batch_size : (i + 1) * self.batch_size
            ] = codes.detach()
            print(f"Iteration: {(i+1):02d} / {n_iter}", end="\r")

        self._save_topk(sampled_codes)

    def _find_topk_not_too_similar(self, sampled_codes, top_k=10):
        """Find the top k codes that are not too similar to each other"""
        topk_search = top_k**2
        topk_indices = torch.argsort(sampled_codes, descending=True)[:topk_search]
        # Check the code values of the topk indices and remove those that are too similar
        topk_codes = sampled_codes[topk_indices]

        # Filter out codes that are too similar
        prev = topk_codes[0]
        filtered_indices = []
        eps = 0.02
        for i in range(1, len(topk_codes)):
            if torch.abs((prev - topk_codes[i])) > eps:
                filtered_indices.append(i)
                prev = topk_codes[i]

        # Take the topk codes if they are smaller than the topk search
        if len(filtered_indices) < top_k:
            topk_indices = filtered_indices[:top_k]
        else:
            topk_indices = torch.argsort(sampled_codes, descending=True)[:top_k]

        return topk_indices

    def _save_topk(self, sampled_codes):
        for j, code in enumerate(sampled_codes.T):
            print(f"Save topk for dim {j:02d}", end="\r")
            # topk_indices = torch.argsort(code, descending=True)[: self.top_k]

            topk_indices = self._find_topk_not_too_similar(code, top_k=self.top_k)
            topk_images, topk_latents = [], []
            for index in topk_indices:
                img, latent = self.dataset[index]
                topk_images.append(img)
                topk_latents.append(latent)

            out_path = os.path.join(self.out_path, f"{j:02d}", "sampled_latents")
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            self._save_latents(out_path, topk_latents, j)

            out_path = os.path.join(self.out_path, f"{j:02d}", "sampled_images")
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            self._save_images(out_path, topk_images, j)

    def _save_latents(self, out_path, topk_latents, j):
        """Save topk latents for each dimension to disk"""
        for k, latent in enumerate(topk_latents):
            latent = latent["arr_0"]
            np.save(os.path.join(out_path, f"sampled_latent_{k:02d}.npy"), latent)

    def _save_images(self, out_path, images, dim):
        """Save topk images for each dimension to disk"""
        fig, axes = plt.subplots(2, 5, figsize=(5, 2))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)
        topk = 10

        for k, img in enumerate(images):
            img = img.permute(1, 2, 0).numpy()
            img = img_to_uint8(img)
            fig_im, ax_im = plt.subplots(1)
            ax_im.axis("off")
            ax_im.imshow(img)
            for ext in ["png", "pdf"]:
                fname = os.path.join(out_path, f"sampled_image_{k:02d}.{ext}")
                fig_im.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close(fig_im)

            if k < topk:
                ax = axes[k // 5, k % 5]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis("off")
                ax.imshow(img)

        out_path = os.path.dirname(out_path)
        for ext in ["png", "pdf"]:
            fname = os.path.join(out_path, f"{dim}_stylegan.{ext}")
            fig.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


class Optimizer(nn.Module):
    """Optimize the latent topk latent codes for selected dimensions"""

    def __init__(
        self,
        lr,
        max_iter,
        dim,
        in_path,
        truncation,
        device,
        alpha=0.15,
        beta=2.0,
        window_size=50,
    ):
        super().__init__()
        self.lr = lr
        self.min_iter = 100
        self.max_iter = max_iter
        self.dim = dim
        self.dim = dim
        self.in_path = in_path
        self.truncation = truncation
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.window_size = window_size
        self.threshold = 2e-4

    def optimize_latents(self, generator, comparator):
        sampled_latents = self._load_latents()
        latent_shape = (len(sampled_latents),) + sampled_latents[0].shape
        optimized_latents = torch.zeros(latent_shape)
        optimized_images = []
        generator.to(self.device)
        comparator.to(self.device)
        for k, latent in enumerate(sampled_latents):
            print(f"Optimization: {(k):02d}\n")
            optimized_latent, optimized_image = self.train(
                latent, generator, comparator
            )
            optimized_latents[k] += optimized_latent.cpu()
            optimized_images.append(optimized_image.cpu().squeeze(0))
        self._save_latents(optimized_latents)
        self._save_images(optimized_images)

    def train(self, sampled_latent, generator, comparator):
        sampled_latent = torch.from_numpy(sampled_latent)
        latent = nn.Parameter(sampled_latent, requires_grad=True)
        optim = torch.optim.Adam([latent], lr=self.lr)
        losses = []
        for param in generator.parameters():
            param.requires_grad = True
        for param in comparator.parameters():
            param.requires_grad = True

        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        for i in range(self.max_iter):
            # Get sparse codes from latent
            img = gen_utils.w_to_img(generator, latent.to(self.device), to_np=False)
            img = torch.clamp(img, min=0, max=255)
            img = img / 255
            img = transforms(img)
            codes = comparator(img, transform=False)[1]

            # Calculate loss, increase the absolute value in a dimension
            # and decrease the NLL in that dimension
            log_code = F.log_softmax(codes, dim=0)[self.dim]
            abs_loss = -self.alpha * codes[self.dim]
            nll = -self.beta * log_code
            loss = abs_loss + nll

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())
            print(
                f"Iteration: {(i+1):02d}, Abs loss: {abs_loss:.3f}, NLL: {nll:.3f}",
                end="\r",
            )
            if (i + 1) % self.window_size == 0 and (i + 1) > self.min_iter:
                window = losses[(i + 1 - self.window_size) : i + 1]
                print("Checking convergence criterion...\n")
                if abs(window[-1]) - abs(window[0]) < self.threshold:
                    print(
                        f"Latent code converged. Stopping optimization after {i+1} iterations.\n"
                    )
                    break

        return latent.data.detach(), img.detach()

    def _load_latents(self):
        in_path = os.path.join(self.in_path, f"{self.dim:02d}", "sampled_latents")
        if not os.path.exists(in_path):
            raise Exception(f"No latent vectors sampled for dimension: {self.dim:02d}")
        sampled_latents = []
        for f in sorted(os.listdir(in_path)):
            if f.endswith(".npy"):
                latent = np.load(os.path.join(in_path, f))
                sampled_latents.append(latent)
        return sampled_latents

    def _save_latents(self, optimized_latents):
        out_path = os.path.join(self.in_path, f"{self.dim:02d}", "optimized_latents")
        out_path = os.path.join(self.in_path, f"{self.dim:02d}", "optimized_latents")
        if not os.path.exists(out_path):
            print(f"Creating directories...\n")
            print(f"Creating directories...\n")
            os.makedirs(out_path)
        for k, latent in enumerate(optimized_latents):
            latent = latent.detach().cpu().numpy()
            np.save(os.path.join(out_path, f"optimized_latent_{k:02d}.npy"), latent)

    def _save_images(self, images):
        out_path = os.path.join(self.in_path, f"{self.dim:02d}")
        if not os.path.exists(out_path):
            print(f"Creating directories...\n")
            print(f"Creating directories...\n")
            os.makedirs(out_path)

        img_path = os.path.join(out_path, "optimized_images")
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        fig, axes = plt.subplots(2, 5, figsize=(5, 2))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.0, hspace=0.0)
        topk = 10
        for k, img in enumerate(images):
            img = img.permute(1, 2, 0).numpy()
            img = img_to_uint8(img)

            fig_im, ax_im = plt.subplots(1)
            ax_im.axis("off")
            ax_im.imshow(img)
            for ext in ["png", "pdf"]:
                fig_im.savefig(
                    os.path.join(img_path, f"optimized_image_{k:02d}.{ext}"),
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0,
                )
            plt.close(fig_im)

            if k < topk:
                ax = axes[k // 5, k % 5]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis("off")
                ax.imshow(img)

        for ext in ["png", "pdf"]:
            fname = os.path.join(out_path, f"{self.dim}_stylegan_optimized.{ext}")
            fig.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.sample_dataset:
        print("Sampling latents...\n")
        generator = StyleGanGenerator(
            out_path="./data/stylegan_dataset_centroids",
            n_samples=args.n_samples,
            truncation=args.truncation,
            batch_size=args.batch_size,
            device=args.device,
        )
        generator.sample_dataset()

    # We can either sample latents again or optimize previously stored ones
    if args.find_topk:
        find_topk_latents(
            args.model_name,
            args.module_name,
            args.embedding_path,
            args.n_samples,
            args.batch_size,
            args.truncation,
            args.top_k,
            args.device,
        )

    if args.optimize_topk:
        for dim in args.dim:
            print("Optimizing dimension: {}".format(dim))
            optimize_latents(
                args.model_name,
                args.module_name,
                args.embedding_path,
                dim,
                args.lr,
                args.max_iter,
                args.truncation,
                args.alpha,
                args.beta,
                args.window_size,
                args.device,
            )
