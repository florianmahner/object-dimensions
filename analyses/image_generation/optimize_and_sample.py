#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import torch
import numpy as np
import os

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn 

from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample    
from image_generation.model import LatentPredictor
from torch.utils.data import DataLoader

class Config:
    model_name = 'vgg16_bn'
    module_name = 'classifier.3'
    elasticnet_path = ""
    embedding_dim = 60 # TODO learn this!

    device = 'cuda:0'

    # training stuff
    max_iter = 200 # number of optimizing iterationd
    lr = 0.1 # learning rate
    rnd_seed = 42

    latent_path = ""

    # sampling stuff
    n_samples = 1_000_000
    batch_size = 256
    truncation = 0.4
    top_k = 10

    sample_latents = True


class Sampler(object):
    """ Generate n latent samples a priori for optimization of latent embeddings 
    We generate n images for this using the pretrained big gan and then select the topk images that maximally 
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

        if not os.path.exists(self.out_path):
            print('Creating directories...\n')
            os.makedirs(self.out_path)

    def sample(self, generator, comparator):
        noise_vector = truncated_noise_sample(truncation=self.truncation, batch_size=self.n_samples)
        dim_vector = truncated_noise_sample(truncation=self.truncation, batch_size=self.n_samples)
        noise_vector = torch.from_numpy(noise_vector)
        dim_vector = torch.from_numpy(dim_vector)

        sampled_latents = torch.cat((noise_vector, dim_vector), dim=1)
        dl = DataLoader(sampled_latents, batch_size=self.batch_size)

        generator.to(self.device)
        comparator.to(self.device)
        generator.eval()
        comparator.eval()

        sampled_codes = torch.zeros(self.n_samples, self.n_dims)
        with torch.no_grad():
            for i, batch in enumerate(dl):
                batch = batch.to(self.device)
                images = generator(batch, self.truncation)
                codes, _ = comparator(images)
                sampled_codes[i*self.batch_size:(i+1)*self.batch_size] += codes.cpu()
                print(f'Iteration: {(i+1):02d}\n')

        self._save_topk(generator, sampled_codes, sampled_latents)

    def _save_topk(self, generator, sampled_codes, sampled_latents):
        for j, code in enumerate(sampled_codes.T):
            topk_indices = torch.argsort(code, descending=True)[:self.top_k]
            topk_latents = sampled_latents[topk_indices]
            self._save_latents(generator, topk_latents, j)

    def _save_latents(self, generator, topk_latents, j):
        out_path = os.path.join(self.out_path, f'{j:02d}', 'sampled_latents')
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        images = []
        for k, latent in enumerate(topk_latents):
            torch.save(latent, os.path.join(out_path, f'sampled_latent_{k:02d}.pt'))
            with torch.no_grad():
                latent = latent.to(self.device)
                img = generator(latent, self.truncation).cpu().squeeze(0)
                images.append(img)
        self._save_images(images, out_path)

    def _save_images(self, images, out_path):
        out_path = os.path.join(out_path, f'images')
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        for k, img in enumerate(images):
            shifted_img = global_shift(img)
            img = img.permute(1, 2, 0).numpy()
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(os.path.join(out_path, f'image_{k:02d}.jpg'))
            plt.clf()
            shifted_img = shifted_img.permute(1, 2, 0).numpy()
            plt.imshow(shifted_img)
            plt.axis('off')
            plt.savefig(os.path.join(out_path, f'shifted_image_{k:02d}.jpg'))
            plt.close()


class Trainer(nn.Module):
    def __init__(self, lr, max_iter, dim, latent_size, in_path, truncation,
                device, alpha=0.15, beta=2.):
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
        self.window_size = 50
        self.threshold = 1e-4

    def optimize_latents(self, generator, comparator):
        sampled_latents = self._load_latents()
        optimized_latents = torch.zeros(len(sampled_latents), self.latent_size)
        optimized_images = []
        generator.to(self.device)
        comparator.to(self.device)
        generator.eval()
        comparator.eval()
        for k, latent in enumerate(sampled_latents):
            print(f'Optimization: {(k+1):02d}\n')
            optimized_latent = self.train(latent, generator, comparator)
            optimized_latents[k] += optimized_latent.cpu()
            optimized_images.append(generator(optimized_latent, self.truncation).detach().cpu().squeeze(0))
        self._save_latents(optimized_latents)
        self._save_images(optimized_images)

    def train(self, sampled_latent, generator, comparator):
        latent = nn.Parameter(sampled_latent, requires_grad=True).to(self.device)
        optim = torch.optim.Adam([latent], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', verbose=True)
        losses = []
        for i in range(self.max_iter):
            optim.zero_grad()
            img = generator(latent, self.truncation)
            codes, _ = comparator(img)
            log_code = F.log_softmax(codes, dim=1).squeeze(0)[self.dim]
            #TODO: figure out why l1 loss does decrease over time
            #l1_loss = self.lmbda * torch.norm(img, p=1) #employed to reduce noise in the image (we don't want to optimize for noisy images)
            abs_loss = -self.alpha * codes.squeeze(0)[self.dim] #increase absolute dimension value (if we don't add this loss term, abs dimension value won't increase)
            nll = -self.beta * log_code #push probability mass in softmax towards dimension for which optimization is perfomed (argmax -> self.dim)
            loss = (abs_loss + nll)
            losses.append(loss.item())
            loss.backward()
            optim.step()
            scheduler.step(loss)
            print('=============================================================')
            print(f'Iteration: {(i+1):02d}, Abs loss: {abs_loss:.3f}, NLL: {nll:.3f}')
            print('=============================================================\n')
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
        out_path = os.path.join(self.in_path, f'{self.dim:02d}', 'optimized_latents', 'images')
        if not os.path.exists(out_path):
            print(f'Creating directories...\n')
            os.makedirs(out_path)
        for k, img in enumerate(images):
            shifted_img = global_shift(img)
            img = img.permute(1, 2, 0).numpy()
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(os.path.join(out_path, f'image_{k:02d}.jpg'))
            plt.clf()
            shifted_img = shifted_img.permute(1, 2, 0).numpy()
            plt.imshow(shifted_img)
            plt.axis('off')
            plt.savefig(os.path.join(out_path, f'shifted_image_{k:02d}.jpg'))
            plt.close()

def global_shift(img:torch.Tensor):
    shifted_img = img.clone()
    shifted_img -= img.min()
    shifted_img /= shifted_img.max()
    return shifted_img


def sample_latents(cfg):
    device = torch.device(cfg.device)
    predictor = LatentPredictor(cfg.embedding_dim, cfg.model_name, cfg.module_name, device, cfg.elasticnet_path)

    gan = BigGAN.from_pretrained('biggan-deep-256')
    generator = gan.generator

    sampler = Sampler(n_samples=cfg.n_samples, n_dims=cfg.embedding_dim, batch_size=cfg.batch_size, truncation=cfg.truncation, 
                      top_k=cfg.top_k, out_path=cfg.latent_path, device=device)
    sampler.sample(generator, predictor)


def optimize_latents(cfg):
    device = torch.device(cfg.device)
    predictor = LatentPredictor(cfg.embedding_dim, cfg.model_name, cfg.module_name, device, cfg.elasticnet_path)

    gan = BigGAN.from_pretrained('biggan-deep-256')
    generator = gan.generator
    trainer = Trainer(lr=cfg.lr, max_iter=cfg.max_iter, dim=cfg.embedding_dim, latent_size=256, 
                      in_path=cfg.latent_path, truncation=cfg.truncation, device=device)
    
    trainer.optimize_latents(generator, predictor)

if __name__ == '__main__':
    cfg = Config()

    np.random.seed(cfg.rnd_seed)
    random.seed(cfg.rnd_seed)
    torch.manual_seed(cfg.rnd_seed)

    # TODO dont make this an if else statement!
    if cfg.sample_latents:
        sample_latents(cfg)

    else:
        optimize_latents(cfg)
