#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import argparse

import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

from PIL import Image

from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--n_samples", type=int, default=1000)
parser.add_argument("--truncation", type=float, default=0.4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--sample_fresh", action="store_true")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(network, dataloader):
    criterion = nn.MSELoss()
    optimizer = nn.optim.Adam(network.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for img, latent in dataloader:
        img = img.to(device)
        latent = latent.to(device)

        output = network(img)

        loss = criterion(output, latent)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())


def sample_images(n_samples, truncation=0.4, seed=0, batch_size=16):
    """
    Sample n images and latents using the pretrained big gan
    """
    noise_vector = truncated_noise_sample(truncation=truncation, batch_size=n_samples)
    dim_vector = truncated_noise_sample(truncation=truncation, batch_size=n_samples)

    gan = BigGAN.from_pretrained("biggan-deep-256")
    generator = gan.generator

    noise_vector = torch.from_numpy(noise_vector)
    dim_vector = torch.from_numpy(dim_vector)

    sampled_latents = torch.cat((noise_vector, dim_vector), dim=1)
    dataloader = DataLoader(sampled_latents, batch_size=batch_size)

    generator.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        generator = torch.nn.DataParallel(generator, device_ids=[0, 1, 2])

    generator.eval()

    n_iter = len(dataloader)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)  # shape (batch_size, 256)
            images = generator(batch, truncation)
            # images = img_to_uint8(images)


class Discriminator(nn.Module):
    """Architecture of the discriminator"""

    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        ndf = 64
        nc = 3
        self.net = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Linear(1, 512, bias=True),
            nn.Sigmoid(),  # NOTE do i want this?!
        )

    def forward(self, input):
        return self.net(input)


class Dataset(torch.utils.data.Dataset):
    """Load a dataset of image (X) and latent pairs (y)"""

    def __init__(self, data_path, img_size=256):
        self.data_path = data_path
        self.transfm = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        data_point = np.load(self.data_path[idx])
        img = Image.fromarray(data_point["img"])
        img = self.transfm(img)

        latent = torch.from_numpy(data_point["latent"])

        return img, latent


if __name__ == "__main__":

    img_size = 256
    img_root = "./data/imagenet_generated"
