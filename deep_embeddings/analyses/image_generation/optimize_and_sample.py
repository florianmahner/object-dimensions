#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import torch
import numpy as np
import os

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn 
import matplotlib.gridspec as gridspec

from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample
from latent_predictor import LatentPredictor
from torch.utils.data import DataLoader

class Config:
    model_name = 'vgg16_bn'
    module_name = 'classifier.3'
    regression_path = "../sparse_codes/sparse_code_predictions"

    device = 'cuda:0'

    # training stuff
    max_iter = 200 # number of optimizing iterationd
    lr = 0.001 # learning rate
    rnd_seed = 42

    dim = 6 # specify a dim to do optimization for!

    latent_path = "./latent_samples"

    # sampling stuff
    n_samples = 100_000
    batch_size = 32
    truncation = 0.4 # truncation values for noise sample
    top_k = 16

    sample_latents = False


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

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            generator = torch.nn.DataParallel(generator, device_ids=[1,0,2])
            comparator = torch.nn.DataParallel(comparator, device_ids=[1,0,2])

        generator.eval()
        comparator.eval()

        sampled_codes = torch.zeros(self.n_samples, self.n_dims).to(self.device)
        sampled_latents = sampled_latents.to(self.device)
        n_iter = len(dl)
        with torch.no_grad():
            for i, batch in enumerate(dl):
                batch = batch.to(self.device) # shape (batch_size, 256)
                images = generator(batch, self.truncation)
                _, codes = comparator(images)
                sampled_codes[i*self.batch_size:(i+1)*self.batch_size] += codes
                print(f'Iteration: {(i+1):02d} / {n_iter}', end='\r')

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

        for k, latent in enumerate(topk_latents):
            torch.save(latent, os.path.join(out_path, f'sampled_latent_{k:02d}.pt'))

        with torch.no_grad():    
            images = generator(topk_latents, self.truncation)
                
        self._save_images(images, out_path)

    def _save_images(self, images, out_path):
        out_path = os.path.join(out_path, f'images')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        images = images.cpu()
        for k, img in enumerate(images):
            img = global_shift(img)
            img = img.permute(1, 2, 0).numpy()
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(os.path.join(out_path, f'image_{k:02d}.jpg'))
            plt.clf()
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
        self.threshold = 2e-4
        

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
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', verbose=True)
        losses = []
        for i in range(self.max_iter):
            optim.zero_grad()
            img = generator(latent, self.truncation) # Create an image
            _, codes = comparator(img) # Extract VGG features
            log_code = F.log_softmax(codes, dim=1).squeeze(0)[self.dim] # Get the value of that image within the dimension we want to optimize for
            
            # Increase absolute dimension value aka dimension size reward (if we don't add this loss term, abs dimension value won't increase)
            abs_loss = -self.alpha * codes.squeeze(0)[self.dim] 
            nll = -self.beta * log_code #push probability mass in softmax towards dimension for which optimization is perfomed (argmax -> self.dim)
            loss = (abs_loss + nll)
            losses.append(loss.item())
            loss.backward()
            optim.step()
            # scheduler.step(loss)
            
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
            img = global_shift(img)
            img = img.permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis('off')

        fname = os.path.join(out_path, f'dim_{self.dim}_optimized.png')
        # fig.suptitle("Dimension: {}".format(self.dim))
        fig.savefig(fname, dpi=150)
        plt.close(fig)

def global_shift(img:torch.Tensor):
    shifted_img = img.clone()
    shifted_img -= img.min()
    shifted_img /= shifted_img.max()
    return shifted_img


def sample_latents(cfg):
    device = torch.device(cfg.device)
    predictor = LatentPredictor(cfg.model_name, cfg.module_name, device, cfg.regression_path)

    gan = BigGAN.from_pretrained('biggan-deep-256')
    generator = gan.generator

    sampler = Sampler(n_samples=cfg.n_samples, n_dims=predictor.embedding_dim, batch_size=cfg.batch_size, truncation=cfg.truncation, 
                      top_k=cfg.top_k, out_path=cfg.latent_path, device=device)
    sampler.sample(generator, predictor)


def optimize_latents(cfg):
    device = torch.device(cfg.device)
    predictor = LatentPredictor(cfg.model_name, cfg.module_name, device, cfg.regression_path)

    gan = BigGAN.from_pretrained('biggan-deep-256')
    generator = gan.generator

    trainer = Trainer(lr=cfg.lr, max_iter=cfg.max_iter, dim=cfg.dim , latent_size=256, 
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
