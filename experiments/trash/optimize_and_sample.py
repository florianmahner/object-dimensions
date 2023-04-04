#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["sample_latents", "optimize_latents"]

import random
import torch
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample
from deep_embeddings.utils.latent_predictor import LatentPredictor
from deep_embeddings.utils.utils import img_to_uint8
from torch.utils.data import DataLoader

from deep_embeddings import ExperimentParser

parser = ExperimentParser(description="BigGAN latent space optimization")
parser.add_argument(
    "--embedding_path",
    type=str,
    default="./weights/params/pruned_q_mu_epoch_300.txt",
    help="Path to weights directory",
)
parser.add_argument(
    "--model_name", type=str, default="vgg16_bn", help="Model to load from THINGSvision"
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
    "--window_size",
    type=int,
    default=50,
    help="Window size of trainer to check convergence of latent dimenisonality",
)
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for sampling")
parser.add_argument(
    "--truncation", type=float, default=0.4, help="Truncation value for noise sample"
)
parser.add_argument("--top_k", type=int, default=16, help="Top k values to sample from")
parser.add_argument(
    "--sample_latents",
    type=str,
    default="False",
    choices=("True", "False"),
    help="Sample latents",
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
    default=1.0,
    help="Weight of the absolute value in a dimension to optimize for",
)
parser.add_argument(
    "--beta",
    type=float,
    default=1.0,
    help="Weight for the softmax loss in the dimension optimization",
)
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")


def global_shift(img: torch.Tensor):
    shifted_img = img.clone()
    shifted_img -= img.min()
    shifted_img /= shifted_img.max()
    return shifted_img


def sample_latents(
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

    assert os.path.exists(
        regression_path
    ), f"Regression path {regression_path} does not exist. Run sparse code predictions \
                                              first before optimizing latents."

    predictor = LatentPredictor(model_name, module_name, device, regression_path)

    gan = BigGAN.from_pretrained("biggan-deep-256")
    generator = gan.generator

    sampler = Sampler(
        n_samples=n_samples,
        n_dims=predictor.embedding_dim,
        batch_size=batch_size,
        truncation=truncation,
        top_k=top_k,
        out_path=out_path,
        device=device,
    )
    sampler.sample(generator, predictor)


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

    gan = BigGAN.from_pretrained("biggan-deep-256")
    generator = gan.generator

    trainer = Optimizer(
        lr=lr,
        max_iter=max_iter,
        dim=dim,
        latent_size=256,
        in_path=latent_path,
        truncation=truncation,
        device=device,
        alpha=alpha,
        beta=beta,
        window_size=window_size,
    )

    trainer.optimize_latents(generator, predictor)


class Sampler(object):
    """Generate n latent samples a priori for optimization of latent embeddings
    We generate n images for this using the pretrained big gan and then select the topk images that maximally
    activate each of the embedding dimension. We then optimize for these topk latents!"""

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

        if not os.path.exists(self.out_path):
            print("Creating directories...\n")
            os.makedirs(self.out_path)

    def sample(self, generator, comparator):
        print("Sampling truncated latents ...")

        noise_vector = truncated_noise_sample(
            truncation=self.truncation, batch_size=self.n_samples
        )
        dim_vector = truncated_noise_sample(
            truncation=self.truncation, batch_size=self.n_samples
        )

        noise_vector = torch.from_numpy(noise_vector)
        dim_vector = torch.from_numpy(dim_vector)

        sampled_latents = torch.cat((noise_vector, dim_vector), dim=1)
        dataloader = DataLoader(sampled_latents, batch_size=self.batch_size)

        generator.to(self.device)
        comparator.to(self.device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            generator = torch.nn.DataParallel(generator, device_ids=[0, 1, 2])
            comparator = torch.nn.DataParallel(comparator, device_ids=[0, 1, 2])

        generator.eval()
        comparator.eval()

        sampled_codes = torch.zeros(self.n_samples, self.n_dims, device=self.device)
        n_iter = len(dataloader)
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                batch = batch.to(self.device)  # shape (batch_size, 256)
                images = generator(batch, self.truncation)
                _, codes = comparator(images)
                sampled_codes[
                    i * self.batch_size : (i + 1) * self.batch_size
                ] += codes.detach()
                print(f"Sampling Iteration: {(i+1):02d} / {n_iter:02d}", end="\r")

        self._save_topk(generator, sampled_codes, sampled_latents)

    def _save_topk(self, generator, sampled_codes, sampled_latents):
        sampled_codes = sampled_codes.cpu()
        for j, code in enumerate(sampled_codes.T):
            print(f"Save topk for dim {j:02d}", end="\r")
            topk_indices = torch.argsort(code, descending=True)[: self.top_k]
            topk_latents = sampled_latents[topk_indices]
            self._save_latents(generator, topk_latents, j)

    def _save_latents(self, generator, topk_latents, j):
        out_path = os.path.join(self.out_path, f"{j:02d}", "sampled_latents")
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        for k, latent in enumerate(topk_latents):
            latent = latent.cpu().numpy()
            torch.save(latent, os.path.join(out_path, f"sampled_latent_{k:02d}.pt"))

        with torch.no_grad():
            topk_latents = topk_latents.to(self.device)
            images = generator(topk_latents, self.truncation)

        self._save_images(images, out_path)

    def _save_images(self, images, out_path):
        out_path = os.path.dirname(out_path)
        out_path = os.path.join(out_path, f"generated_images")
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        images = images.cpu()
        for k, img in enumerate(images):
            img = global_shift(img)
            img = img.permute(1, 2, 0).numpy()
            plt.imshow(img)
            plt.axis("off")
            plt.savefig(
                os.path.join(out_path, f"image_{k:02d}_generated.png"),
                bbox_inches="tight",
                dpi=300,
                pad_inches=0,
            )
            plt.clf()
            plt.close()


class Optimizer(nn.Module):
    def __init__(
        self,
        lr,
        max_iter,
        dim,
        latent_size,
        in_path,
        truncation,
        device,
        alpha=0.15,
        beta=2.0,
        window_size=50,
    ):
        super().__init__()
        self.lr = lr
        self.min_iter = 200
        self.max_iter = max_iter
        self.dim = dim
        self.latent_size = latent_size
        self.in_path = in_path
        self.truncation = truncation
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.window_size = window_size
        self.threshold = 2e-4

    def optimize_latents(self, generator, comparator):
        sampled_latents = self._load_latents()
        optimized_latents = torch.zeros(len(sampled_latents), self.latent_size)
        optimized_images = []
        generator.to(self.device)
        comparator.to(self.device)

        # We optimizer here the sampled latents for all topk images that a priori maximally activated a dimension
        for k, latent in enumerate(sampled_latents):
            print(f"Optimization: {(k):02d}\n")
            optimized_latent = self.train(latent, generator, comparator)
            optimized_latents[k] += optimized_latent.cpu()
            optimized_images.append(
                generator(optimized_latent, self.truncation).detach().cpu().squeeze(0)
            )
        self._save_latents(optimized_latents)
        self._save_images(optimized_images)

    def train(self, sampled_latent, generator, comparator):
        sampled_latent = torch.from_numpy(sampled_latent)
        latent = nn.Parameter(sampled_latent, requires_grad=True)
        optim = torch.optim.Adam([latent], lr=0.002)
        latent = latent.to(self.device)
        losses = []
        for i in range(self.max_iter):
            optim.zero_grad()
            img = generator(latent, self.truncation)  # Create an image
            _, codes = comparator(img)  # Predict sparse codes for that image
            log_code = F.log_softmax(codes, dim=0)[
                self.dim
            ]  # Get the value of that image within the dimension we want to optimize for

            # Increase absolute dimension value aka dimension size reward (if we don't add this loss term, abs dimension value won't increase)
            # TODO Alpha and Beta are hyperparameters  -> How and why do we set them like this?
            # abs_loss = -self.alpha * codes[self.dim]
            # nll = -self.beta * log_code #push probability mass in softmax towards dimension for which optimization is perfomed (argmax -> self.dim)
            # loss = (abs_loss + nll)

            # Increase the value of the dimension we want to optimize for
            abs_loss = -self.alpha * codes[self.dim]
            nll = 0
            loss = abs_loss
            losses.append(loss.item())
            loss.backward()
            optim.step()
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

        return latent.data.detach()

    def _load_latents(self):
        in_path = os.path.join(self.in_path, f"{self.dim:02d}", "sampled_latents")
        if not os.path.exists(in_path):
            raise Exception(f"No latent vectors sampled for dimension: {self.dim:02d}")
        sampled_latents = []
        for f in sorted(os.listdir(in_path)):
            if f.endswith("pt"):
                sampled_latents.append(
                    torch.load(os.path.join(in_path, f), map_location=self.device)
                )
        return sampled_latents

    def _save_latents(self, optimized_latents):
        out_path = os.path.join(self.in_path, f"{self.dim:02d}", "optimized_latents")
        if not os.path.exists(out_path):
            print(f"Creating directories...\n")
            os.makedirs(out_path)
        for k, latent in enumerate(optimized_latents):
            torch.save(latent, os.path.join(out_path, f"optimized_latent_{k:02d}.pt"))

    def _save_images(self, images):
        out_path = os.path.join(self.in_path, f"{self.dim:02d}", "optimized_images")
        if not os.path.exists(out_path):
            print(f"Creating directories...\n")
            os.makedirs(out_path)
        for k, img in enumerate(images):
            img = global_shift(img)
            img = img.permute(1, 2, 0).numpy()
            plt.imshow(img)
            plt.axis("off")
            plt.savefig(
                os.path.join(out_path, f"image_{k:02d}_optimized.png"),
                bbox_inches="tight",
                dpi=300,
                pad_inches=0,
            )
            plt.clf()
            plt.close()


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # We can either sample latents again or optimize previously stored ones
    if args.sample_latents:
        print("Sampling latents...\n")
        sample_latents(
            args.model_name,
            args.module_name,
            args.embedding_path,
            args.n_samples,
            args.batch_size,
            args.truncation,
            args.top_k,
            args.device,
        )

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
