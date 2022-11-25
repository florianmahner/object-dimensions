#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import argparse
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_images(n_samples, truncation=0.4, seed=0, batch_size=16):
    """ 
    Sample n images and latents using the pretrained big gan
    """
    noise_vector = truncated_noise_sample(truncation=truncation, batch_size=n_samples)
    dim_vector = truncated_noise_sample(truncation=truncation, batch_size=n_samples)

    gan = BigGAN.from_pretrained('biggan-deep-256')
    generator = gan.generator

    
    noise_vector = torch.from_numpy(noise_vector)
    dim_vector = torch.from_numpy(dim_vector)

    sampled_latents = torch.cat((noise_vector, dim_vector), dim=1)
    dataloader = DataLoader(sampled_latents, batch_size=batch_size)

    generator.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        generator = torch.nn.DataParallel(generator, device_ids=[0,1,2])

    generator.eval()

    n_iter = len(dataloader)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = batch.to(device) # shape (batch_size, 256)
            images = generator(batch, truncation)
            # images = img_to_uint8(images)
            



if __name__ == '__main__':
    pass