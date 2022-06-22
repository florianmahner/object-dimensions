
'''

a) Curate a list of paiwise similarity indices
b)  Loss in terms of pairwise similarities

Write a dataloader still for this!

1. Take two samples from S (dot product)
2. Take two embedding vector

3. Loss based on the distance of the embedding vectors to the true one!

Q. How can I get this across multiple batches?

I still need a prior then to evaluate the model!

Also: place the model on gpu!
'''
from typing import DefaultDict
import numpy as np
from vi_embeddings.model import VI, GaussianPrior
from itertools import combinations
import torch.nn.functional as F
import torch
import random
from vi_embeddings.utils import compute_rsm, correlate_rsms

class PairwiseDataset(torch.utils.data.Dataset):
    ''' Sample pairwise indices from the list combinations'''
    def __init__(self, pairwise_indices):
        super().__init__()
        self.pairwise_indices = pairwise_indices
        self.n_indices = len(pairwise_indices)

    def __getitem__(self, idx):
        return self.pairwise_indices[idx]

    def __len__(self):
        return self.n_indices



device = torch.device('cuda:0')
torch.manual_seed(42)
np.random.seed(42)

features = np.load('/LOCAL/fmahner/THINGS/vgg_features6/features.npy')
n_objects, n_features = features.shape
init_dim = 50

features = (features - np.mean(features, axis=0)) / np.std(features, axis=0) # zscore features
features = (features * init_dim) / n_features # NOTE check why we do this?!
similarity_mat = features @ features.T 
similarity_mat = torch.from_numpy(similarity_mat).to(device)


indices = list(combinations(range(n_objects), 2))[:30_000]
random.shuffle(indices)

n_pairwise = len(indices)


model = VI(n_objects, init_dim)
model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.001)

dataset = PairwiseDataset(indices)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
prior = GaussianPrior(n_objects, init_dim)
prior.to(device)

n_batches = len(dataloader)


epochs = 100

from collections import defaultdict
outputs = defaultdict(list)


for ep in range(epochs):
    losses = []
    for k, batch in enumerate(dataloader):
        batch = [i.to(device) for i in batch]

        optim.zero_grad()

        sim_features = similarity_mat[batch]
        embedding, loc, scale = model()

        i, j = batch
        embeddings_i = F.relu(embedding[i])# NOTE Lukas makes a positivity constraint on these logits, ie. F.relu(embeddings). Does that make sense when drawn from unit gaussian?
        embeddings_j = F.relu(embedding[j])

        sim_embedding = torch.sum(embeddings_i * embeddings_j, dim=1) # NOTE dot product of the embedding!
        likelihood = F.mse_loss(sim_features, sim_embedding)
        likelihood /= init_dim # NOTE same as with similarity mat, check why we do this?!

        # log probability of variational distribution
        log_q = prior.normalized_pdf(embedding, loc, scale).log()
        
        # gaussian prior log probability
        log_p = prior(embedding).log()

        complexity_loss = (1 / n_pairwise) * (log_q.sum() - log_p.sum())
        loss = likelihood + complexity_loss

        loss.backward()
        optim.step()

        losses.append(loss.item())

        print(f'Batch {k}/{n_batches} MSE {loss.item()}', end='\r')
        i += 1

    print(f'Average loss epoch {ep}: {np.mean(losses)}')

    if (ep ) % 10 == 0:
        embedding = model.detached_params()['q_mu']
        np.savetxt(f'./weightstest/weights_epoch_{ep+1}.txt', embedding)

        rsm_embedding = compute_rsm(embedding)
        rsm_features = compute_rsm(features)
        corr = correlate_rsms(rsm_features, rsm_embedding)
        print(f'Correlation between RDMs at epoch {ep+1}: {corr}')



import thingsvision.vision as vision
vision.plot_rdm('./rdm_vgg', features)
vision.plot_rdm('./rdm_embedding', embedding)

     



