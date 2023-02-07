#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["sample_latents", "optimize_latents"]

import torchvision
import argparse

import random
import torch
import os
import glob

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn 
import numpy as np
import matplotlib.gridspec as gridspec


# NOTE Havent found a better solution right now. Alternative would be to make the repo pip installable, which
# NOTE would require me to change the code base.
# import sys
# sys.path.append("./deep_embeddings/analyses/image_generation/stylegan_xl")
# from deep_embeddings.analyses.image_generation.stylegan_xl_previous import legacy
# from deep_embeddings.analyses.image_generation.stylegan_xl_previous.dnnlib import util
# from deep_embeddings.analyses.image_generation.stylegan_xl_previous.torch_utils import gen_utils
# from deep_embeddings.analyses.image_generation.stylegan_xl_previous.gen_images import make_transform

import sys
sys.path.append("./stylegan_xl")

from stylegan_xl import legacy
from stylegan_xl.dnnlib import util
from stylegan_xl.torch_utils import gen_utils
from stylegan_xl.gen_images import make_transform


from deep_embeddings.utils.utils import img_to_uint8
from deep_embeddings import ExperimentParser
from deep_embeddings.utils.latent_predictor import LatentPredictor
from torch.utils.data import DataLoader
from PIL import Image

parser = ExperimentParser(description="Optimize and sample StyleGAN")
parser.add_argument("--embedding_path", type=str, default="./weights/params/pruned_q_mu_epoch_300.txt", help="Path to weights directory")
parser.add_argument("--model_name", type=str, default="vgg16_bn", help="Model to load from THINGSvision")
parser.add_argument("--module_name", type=str, default="classifier.3", help="Layer of the model to load from THINGSvision")
parser.add_argument("--n_samples", type=int, default=2_000_000, help="Number of latent samples to generate")
parser.add_argument("--window_size", type=int, default=50, help="Window size of trainer to check convergence of latent dimenisonality")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size for sampling")
parser.add_argument("--truncation", type=float, default=0.4, help="Truncation value for noise sample")
parser.add_argument("--top_k", type=int, default=16, help="Top k values to sample from")
parser.add_argument("--sample_dataset", type=str, default="True", choices=("True", "False"), help="Sample entire dataset")
parser.add_argument("--find_topk", type=str, default="False", choices=("True", "False"), help="Find top k latent dimensions")
parser.add_argument("--max_iter", type=int, default=200, help="Number of optimizing iterations")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--dim", type=int, default=(1,2), nargs="+", help="Dimension to optimize for")
parser.add_argument("--alpha", type=float, default=1.0, help="Weight of the absolute value in a dimension to optimize for")
parser.add_argument("--beta", type=float, default=1.0, help="Weight for the softmax loss in the dimension optimization")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")


transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=256),
    torchvision.transforms.CenterCrop(size=(224, 224)),
    torchvision.transforms.PILToTensor()])


def load_style_gan():
    # Load the generator from style gan xl
    network_pkl = "https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet1024.pkl"
    print('Loading networks from "%s"...' % network_pkl)

    with util.open_url(network_pkl) as f:
        generator = legacy.load_network_pkl(f)['G_ema']
        generator = generator.eval().requires_grad_(False)

    return generator

def find_topk_latents(model_name, module_name, embedding_path, n_samples, batch_size, truncation, top_k, device):
    device = torch.device(device)
    base_path = os.path.dirname(os.path.dirname(embedding_path))
    regression_path = os.path.join(base_path, "analyses", "sparse_codes")
    out_path = os.path.join(base_path, "analyses", "per_dim")

    if not os.path.exists(regression_path):
        raise FileNotFoundError(f"Regression path {regression_path} does not exist. Run sparse code predictions \
                                              first before optimizing latents.")

    latent_predictor = LatentPredictor(model_name, module_name, device, regression_path)
    predictor = SparseCodesPredictor(n_samples=n_samples, n_dims=latent_predictor.embedding_dim, batch_size=batch_size, truncation=truncation, 
                      top_k=top_k, out_path=out_path, device=device)
    predictor.predict_latent(latent_predictor)

def optimize_latents(model_name, module_name, embedding_path, dim, lr, max_iter, truncation, alpha, beta, window_size, device):
    base_path = os.path.dirname(os.path.dirname(embedding_path))
    regression_path = os.path.join(base_path, "analyses", "sparse_codes")
    latent_path = os.path.join(base_path, "analyses", "per_dim")

    device = torch.device(device)
    predictor = LatentPredictor(model_name, module_name, device, regression_path)

    generator = load_style_gan()

    trainer = Optimizer(lr=lr, max_iter=max_iter, dim=dim, 
                        in_path=latent_path, truncation=truncation, device=device, 
                        alpha=alpha, beta=beta, window_size=window_size)
    
    trainer.optimize_latents(generator, predictor)


class StyleGanGenerator(object):
    """ Generator class for StyleGan that create a latent and image dataset pair and saves to disk """

    def __init__(self, out_path, n_samples, truncation, batch_size, device):
        self.n_samples = n_samples
        self.out_path = out_path
        self.truncation = truncation
        self.device = device
        self.batch_size = batch_size

        self.generator = load_style_gan()
        self.generator = self.generator.to(self.device)

    def sample_latents(self):
        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        translate = (0,0)
        rotate = 0
        if hasattr(self.generator.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            self.generator.synthesis.input.transform.copy_(torch.from_numpy(m))

        sampled_latents = torch.zeros((self.n_samples, 39, 512))
        seeds = np.random.randint(0, 2**32 - 1, size=self.n_samples)
        for i in range(self.n_samples):
            print(f"Generating latent w_space vector: {i+1}/{self.n_samples} ...", end="\r")

            # Centroids path enbables multi-modal truncation for imagenet generations!
            w_latent = gen_utils.get_w_from_seed(self.generator, 1, self.device, self.truncation, seed=seeds[i], 
                                                centroids_path=None,
                                                class_idx=None)

            sampled_latents[i] = w_latent

        return sampled_latents

    def sample_dataset(self, sampled_latents=None):
        """ Saves all images and latents to disc """
        if not os.path.exists(self.out_path):
            print(f"Creating directory {self.out_path}")
            os.makedirs(os.path.join(self.out_path, "images"))
            os.makedirs(os.path.join(self.out_path, "latents"))

        if not sampled_latents:
            sampled_latents = self.sample_latents()

        dataloader = DataLoader(sampled_latents, batch_size=self.batch_size)
        print('Generating images and sparse code predictions ...\n')

        n_iter = len(dataloader)
        n_samples = 0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                print("Process Batch {}/{}".format(i+1, n_iter), end="\r")
                batch = batch.to(self.device) 
                images = gen_utils.w_to_img(self.generator, batch, to_np=True)

                for latent, img in zip(batch, images):
                    img = Image.fromarray(img, mode="RGB")
                    img.save(os.path.join(self.out_path, "images", f"{n_samples}.jpg"))
                    latent = latent.cpu().numpy()
                    np.save(os.path.join(self.out_path, "latents", f"{n_samples}.npy"), latent)
                    n_samples += 1


class StyleGanDataset(torch.utils.data.Dataset):
    def __init__(self, base_path="./data/stylegan_dataset", transform=None):
        self.latent_paths = glob.glob(os.path.join(base_path, "latents", "*.npy"))
        self.image_paths = glob.glob(os.path.join(base_path, "images", "*.jpg"))

        self.latent_paths.sort()
        self.image_paths.sort()

        self.latents = [np.load(latent_path) for latent_path in self.latent_paths]

        assert len(self.latent_paths) == len(self.image_paths), "Latent and image paths do not match"
        self.transform = transform
        
    def __len__(self):
        return len(self.latent_paths)

    def __getitem__(self, idx):
        # latent = self.latents[idx]
        latent = np.load(self.latent_paths[idx])
        image = Image.open(self.image_paths[idx])
        image = image.convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return image, latent


class SparseCodesPredictor(object):
    """ Generate n latent samples a priori for optimization of latent embeddings 
    We generate n images for this using the pretrained style gan and then select the topk images that maximally 
    activate each of the embedding dimension. We then optimize for these topk latents! """
    def __init__(self, n_samples, n_dims, batch_size, truncation, top_k,
                 out_path, device):
        self.n_samples = n_samples
        self.n_dims = n_dims
        self.batch_size = batch_size
        self.top_k = top_k
        self.truncation = truncation
        self.out_path = out_path
        self.device = device
        self.dataset = StyleGanDataset(base_path="./data/stylegan_dataset", transform=transforms)

        if not os.path.exists(self.out_path):
            print('Creating directories...\n')
            os.makedirs(self.out_path)

    def predict_latent(self, comparator):  
        """ Sample latent using StyleGAN Xl batch wise and generate sparse code predictions for all images.
        Store the top k latents and images for each dimension to disk. """
        comparator.to(self.device)

        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        sampled_codes = torch.zeros((self.n_samples, self.n_dims))
        n_iter = len(dataloader)
        for i, (img, _) in enumerate(dataloader):
            img = img.to(self.device)
            _, codes = comparator(img)
            sampled_codes[i*self.batch_size:(i+1)*self.batch_size] = codes
            print(f'Iteration: {(i+1):02d} / {n_iter}', end='\r')

        self._save_topk(sampled_codes)

    def _save_topk(self, sampled_codes):
        for j, code in enumerate(sampled_codes.T):
            print(f"Save topk for dim {j:02d}", end="\r")
            topk_indices = torch.argsort(code, descending=True)[:self.top_k]

            topk_images, topk_latents = [], []
            for index in topk_indices:
                img, latent = self.dataset[index]
                topk_images.append(img)
                topk_latents.append(latent)            

            out_path = os.path.join(self.out_path, f'{j:02d}', 'sampled_latents')
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            self._save_latents(out_path, topk_latents, j)
            self._save_images(out_path, topk_images, j)

    def _save_latents(self, out_path, topk_latents, j):
        """ Save topk latents for each dimension to disk"""
        for k, latent in enumerate(topk_latents):
            np.save(os.path.join(out_path, f'sampled_latent_{k:02d}.npy'), latent)

    def _save_images(self, out_path, images, dim):
        """ Save topk images for each dimension to disk """
        fig = plt.figure(figsize=(4,4))
        gs1 = gridspec.GridSpec(4,4)
        gs1.update(wspace=0.002, hspace=0.002) # set the spacing between axes.
        for k, img in enumerate(images):
            ax = plt.subplot(gs1[k])
            img = img.permute(1,2,0).numpy()
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.patch.set_edgecolor('black')  

        fname = os.path.join(out_path, f'dim_{dim}_sampled.png')
        fig.savefig(fname, dpi=300)
        plt.close(fig)

class Optimizer(nn.Module):
    def __init__(self, lr, max_iter, dim, in_path, truncation,
                device, alpha=0.15, beta=2., window_size=50):
        super().__init__()
        self.lr = lr
        self.min_iter = 200
        self.max_iter = max_iter
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
        generator.eval()
        comparator.eval()
        for k, latent in enumerate(sampled_latents):
            print(f'Optimization: {(k):02d}\n')
            optimized_latent = self.train(latent, generator, comparator)
            optimized_latents[k] += optimized_latent.cpu()

            img = gen_utils.w_to_img(generator, optimized_latent, to_np=False).detach().cpu().squeeze(0)
            optimized_images.append(img)

            # optimized_images.append(gen÷erator(optimized_latent, self.truncation).detach().cpu().squeeze(0))
        self._save_latents(optimized_latents)
        self._save_images(optimized_images)
        

    def train(self, sampled_latent, generator, comparator):
        sampled_latent = torch.from_numpy(sampled_latent)
        latent = nn.Parameter(sampled_latent, requires_grad=True)
        optim = torch.optim.Adam([latent], lr=self.lr)
        latent = latent.to(self.device)
        
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', verbose=True)
        losses = []
        for i in range(self.max_iter):                
            img = gen_utils.w_to_img(generator, latent, to_np=False)
            img = img.clamp(0, 255) 
            img = img_to_uint8(img)
            _, codes = comparator(img) # Extract features
            codes = codes.squeeze(0)
            log_code = F.log_softmax(codes, dim=0)[self.dim] # Get the value of that image within the dimension we want to optimize for

            # Increase absolute dimension value aka dimension size reward (if we don't add this loss term, abs dimension value won't increase)
            # TODO Alpha and Beta are hyperparameters  -> How and why do we set them like this?
            abs_loss = -self.alpha * codes[self.dim] 
            nll = -self.beta * log_code #push probability mass in softmax towards dimension for which optimization is perfomed (argmax -> self.dim)
            loss = (abs_loss + nll)
    
            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())
            print(f'Iteration: {(i+1):02d}, Abs loss: {abs_loss:.3f}, NLL: {nll:.3f}', end='\r')
            
            if (i+1) % self.window_size == 0 and (i+1) > self.min_iter:
                window = losses[(i+1-self.window_size):i+1]
                print('Checking convergence criterion...\n')
                if abs(window[-1]) - abs(window[0]) < self.threshold:
                    print(f'Latent code converged. Stopping optimization after {i+1} iterations.\n')
                    break

        return latent.data.detach()

    def _load_latents(self):
        in_path = os.path.join(self.in_path, f'{self.dim:02d}', 'sampled_latents')
        if not os.path.exists(in_path):
            raise Exception(f'No latent vectors sampled for dimension: {self.dim:02d}')
        sampled_latents = []
        for f in sorted(os.listdir(in_path)):
            if f.endswith('pt'):
                sampled_latents.append(torch.load(os.path.join(in_path, f), map_location=self.device))
        return sampled_latents

    def _save_latents(self, optimized_latents):
        out_path = os.path.join(self.in_path, f'{self.dim:02d}', 'optimized_latents')
        if not os.path.exists(out_path):
            print(f'Creating directories...\n')
            os.makedirs(out_path)
        for k, latent in enumerate(optimized_latents):
            torch.save(latent, os.path.join(out_path, f'optimized_latent_{k:02d}.pt'))

    def _save_images(self, images):
        out_path = os.path.join(self.in_path, f'{self.dim:02d}', 'optimized_latents')
        if not os.path.exists(out_path):
            print(f'Creating directories...\n')
            os.makedirs(out_path)

        fig = plt.figure(figsize=(4,4))
        gs1 = gridspec.GridSpec(4,4)
        gs1.update(wspace=0.002, hspace=0.002) # set the spacing between axes.
        for k, img in enumerate(images):
            ax = plt.subplot(gs1[k])
            img = img_to_uint8(img)
            img = img.permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.patch.set_edgecolor('black')  
        fname = os.path.join(out_path, f'dim_{self.dim}_optimized.png')
        fig.savefig(fname, dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.sample_dataset == "True":
        print("Sampling latents...\n")
        generator = StyleGanGenerator(out_path="./data/stylegan_dataset", n_samples=args.n_samples, truncation=args.truncation, 
                                           batch_size=args.batch_size, device=args.device)
        generator.sample_dataset()    


    # We can either sample latents again or optimize previously stored ones
    if args.find_topk == "True":
        find_topk_latents(args.model_name, args.module_name,  args.embedding_path, args.n_samples, 
                       args.batch_size, args.truncation, args.top_k, args.device)

    # for dim in args.dim:
    #     print("Optimizing dimension: {}".format(dim))
    #     optimize_latents(args.model_name, args.module_name, args.embedding_path, dim, args.lr,
    #                     args.max_iter, args.truncation, args.alpha, args.beta, args.window_size, args.device)