import numpy as np

import torch
from torchnmf.nmf import NMF


F = np.load("./data/triplets/vgg16_bn/classifier.3/features.npy")

F = torch.from_numpy(F)
F = F.clip(0)
F = F.cuda().t()

model = NMF(F.shape, rank=100)
model = model.cuda()

model.fit(F, max_iter=1000, verbose=True)

dims = model.W
dims = dims.cpu().detach().numpy()

np.savetxt("./data/triplets/vgg16_bn/classifier.3/nmf_dims.txt", dims)

