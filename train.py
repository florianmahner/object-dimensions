import numpy as np
from vi_embeddings.model import VI, SpikeSlabPrior, GaussianPrior
from itertools import combinations
import torch.nn.functional as F
import torch
import random
from vi_embeddings.utils import compute_rsm, correlate_rsms, normalized_pdf, cosine_similarity

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

features = np.load('/LOCAL/fmahner/THINGS/vgg_bn_features6/features.npy')
n_objects, n_features = features.shape
init_dim = 100

features = (features - np.mean(features, axis=0)) / np.std(features, axis=0) # zscore features
features = (features * init_dim) / n_features # NOTE check why we do this?!
# features += 2 # shift the mean of the gaussian distribution to be positive!

similarity_mat = features @ features.T
# similarity_mat /= n_objects**2
similarity_mat = torch.from_numpy(similarity_mat).to(device)
# similarity_mat = F.relu(similarity_mat)

indices = list(combinations(range(n_objects), 2))[:1_000_000]
random.shuffle(indices)

n_pairwise = len(indices)

model = VI(n_objects, init_dim)
model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.001)

dataset = PairwiseDataset(indices)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4)
# prior = GaussianPrior(n_objects, init_dim)  
prior = SpikeSlabPrior(n_objects, init_dim, spike=0.25, slab=1.0, pi=0.5)
prior.to(device)

n_batches = len(dataloader)


epochs = 100
latent_dimensions = []  
for ep in range(epochs):
    model.train()
    losses = []
    for k, batch in enumerate(dataloader):
        batch = [i.to(device) for i in batch]

        optim.zero_grad()
        sim_features = similarity_mat[batch]
        embedding, loc, scale = model()

        i, j = batch
        # NOTE Lukas makes a positivity constraint on these logits, ie. F.relu(embeddings). 
        # NOTE Does that make sense when drawn from unit gaussian?
        
        # embeddings_i = F.relu(embedding[i])
        # embeddings_j = F.relu(embedding[j])

        embeddings_i = embedding[i]
        embeddings_j = embedding[j]

        # There is stil something wrong with the likelihood maybe?!
        sim_embedding = torch.sum(embeddings_i * embeddings_j, dim=1) # NOTE dot product of the embedding!
        # sim_embedding /= init_dim
        # sim_embedding /= init_dim ** 2


        # likelihood = F.cosine_similarity(sim_embedding.unsqueeze_(0), sim_features.unsqueeze_(0))
        # likelihood *= init_dim


        likelihood = F.mse_loss(sim_features, sim_embedding)
        # likelihood /= (n_objects * init_dim)


        # likelihood /= init_dim # NOTE same as with similarity mat, check why we do this?!
        # likelihood *= 10 # This is very hacky -> Find a good solutio to this and check what the complexity loss does?!

        # log probability of variational distribution
        log_q = normalized_pdf(embedding, loc, scale).log()
        
        # gaussian prior log probability of the spike and slab!
        log_p = prior(embedding).log()

        complexity_loss = (1 / n_pairwise) * (log_q.sum() - log_p.sum())
        loss = likelihood + complexity_loss

        loss.backward()
        optim.step()

        losses.append(loss.item())
        print(f'Batch {k}/{n_batches} Likelihood {likelihood.item()}, Complexity {complexity_loss.item()}', end='\r')


    # NOTE Prune stuff -> This doesnt change the embedding dim though out of the box!
    # NOTE The complexity loss is so much higher than the LL at the moment. Therefore weights are pushed to converge
    # NOTE to have zero mean, which seems very off!
    signal, _, _ = model.prune_dimensions(alpha=0.05)    
    dimensionality = signal.shape[0]
    latent_dimensions.append(dimensionality)

    print(f'\nAverage loss epoch {ep}: {np.mean(losses)}, Dimensionality {dimensionality}')


    if (ep ) % 1 == 0:
        embedding = model.detached_params()['q_mu']
        np.savetxt(f'./weights/weights_epoch_{ep+1}.txt', embedding)
        rsm_embedding = compute_rsm(embedding)
        rsm_features = compute_rsm(features)
        corr = correlate_rsms(rsm_features, rsm_embedding)
        print(f'Correlation between RDMs at epoch {ep+1}: {corr}')



import thingsvision.vision as vision
vision.plot_rdm('./rdm_vgg', features)
vision.plot_rdm('./rdm_embedding', embedding)

     



