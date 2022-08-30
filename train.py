import numpy as np
from vi_embeddings.model import VI
from vi_embeddings.priors import SpikeSlabPrior, ExponentialPrior, WeibullPrior
import torch.nn.functional as F
import torch
import random
from vi_embeddings.utils import compute_rsm, correlate_rsms, normalized_pdf, compute_positive_rsm
import os
from copy import deepcopy


# Put argparse for different scripts in difereent files / directories maybe?
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

# NOTE maybe change this to without ref!
features = np.load('/LOCAL/fmahner/THINGS/vgg_bn_features6/features.npy')
n_objects, n_features = features.shape
init_dim = 100


# features = (features - np.mean(features, axis=0)) / np.std(features, axis=0) # zscore features
# features = (features * init_dim) / n_features # NOTE check why we do this?, this gives the best results tho imho

features = np.maximum(features, 0) # we activate beforehand, since all other things are noise, this does really not work...

similarity_mat = features @ features.T
# similarity_mat = np.maximum(similarity_mat, 0)
similarity_mat = torch.from_numpy(similarity_mat).to(device)
# similarity_mat /= n_features # dot product over n features and therefore divide by the same number!

# maybe also do:
similarity_mat = (similarity_mat * init_dim) / n_features
# similarity_mat = (similarity_mat * 10) / n_features
# similarity_mat /= n_features


# Define model path and checkpoints quickly!
# log_path = './weights_exp6_featact'
log_path = './weights_sslab6_norelu_ll_10'
model_name = 'model_epoch_9.tar'
model_path = os.path.join(log_path, model_name)
load_model = False

if not os.path.exists(log_path):
    os.mkdir(log_path)


indices = torch.combinations(torch.arange(n_objects), 2, with_replacement=False)
rand_ind = torch.randperm(len(indices))
indices = indices[rand_ind]


n_pairwise = len(indices)

model = VI(n_objects, init_dim)
model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=0.001)

if load_model:
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optim_state_dict'])
    start_epoch = checkpoint['epoch'] + 1 # we want to resume at next epoch!

else:
    start_epoch = 1


dataset = PairwiseDataset(indices)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8196*8, shuffle=True, num_workers=4, pin_memory=True)


prior = SpikeSlabPrior(n_objects, init_dim, spike=0.25, slab=1.0, pi=0.5)
# prior = ExponentialPrior(n_objects, lmbda=1.0)
# prior = WeibullPrior(lmbda=1.0, k=0.5)
prior.to(device)

n_batches = len(dataloader)

epochs = 2000
latent_dimensions = []  
for ep in range(start_epoch, start_epoch + epochs):
    model.train()
    losses = []
    for k, batch in enumerate(dataloader):

        batch = batch.to(device)
        batch = batch.unbind(1) # convert into two lists

        optim.zero_grad()
        sim_features = similarity_mat[batch]

        embedding, loc, scale = model()

        i, j = batch
        
        embeddings_i = embedding[i]
        embeddings_j = embedding[j]


        # NOTE maybe this was really my stupidity mistake!!!
        # embeddings_i = F.relu(embedding[i])
        # embeddings_j = F.relu(embedding[j])


        # There is stil something wrong with the likelihood maybe?!
        sim_embedding = torch.sum(embeddings_i * embeddings_j, dim=1) # NOTE dot product of the embedding!

        # likelihood = F.cosine_similarity(sim_embedding.unsqueeze_(0), sim_features.unsqueeze_(0))
        # likelihood *= init_dim

        log_likelihood = F.mse_loss(sim_features, sim_embedding)
        # log_likelihood *= init_dim # NOTE added this to balance the loss!!, which seems to work very well!!!!!!!

        # log probability of variational distribution
        log_q = normalized_pdf(embedding, loc, scale).log()
        
        # gaussian prior log probability of the spike and slab!
        log_p = prior(embedding).log()

        complexity_loss = (1 / n_pairwise) * (log_q.sum() - log_p.sum())
        loss = log_likelihood + complexity_loss

        loss.backward()
        optim.step()

        losses.append(loss.item())
        print(f'Batch {k}/{n_batches} Likelihood {log_likelihood.item()}, Complexity {complexity_loss.item()}', end='\r')
        # print(f'Sum {log_q.sum()}, {log_p.sum()}', end='\r')


    # NOTE Prune stuff -> This doesnt change the embedding dim though out of the box!
    # NOTE The complexity loss is so much higher than the LL at the moment. Therefore weights are pushed to converge
    # NOTE to have zero mean, which seems very off!
    signal, pruned_loc, pruned_scale = model.prune_dimensions(alpha=0.05)    
    dimensionality = signal.shape[0]
    latent_dimensions.append(dimensionality)

    print(f'\nAverage loss epoch {ep}: {np.mean(losses)}, Dimensionality {dimensionality}')


    if (ep ) % 50 == 0:
        embedding = model.detached_params()['q_mu']
        np.savetxt(log_path + f'/weights_epoch_{ep}.txt', embedding)
        np.savetxt(log_path + f'/weights_pruned_epoch_{ep}.txt', pruned_loc)
        np.savetxt(log_path + f'/scale_pruned_epoch_{ep}.txt', pruned_scale)
        
        rsm_embedding = compute_rsm(embedding)
        rsm_features = compute_positive_rsm(features)
        corr = correlate_rsms(rsm_features, rsm_embedding)
        print(f'Correlation between RDMs at epoch {ep}: {corr}')


    if (ep % 500) == 0:
        # save model!
        m_path = os.path.join(log_path, 'model_epoch_{}.tar')
        torch.save({'model_state_dict': deepcopy(model.state_dict()),
                    'optim_state_dict': deepcopy(optim.state_dict()),
                    'train_loader': dataloader, 
                    'epoch': ep}, m_path.format(ep))


import thingsvision.vision as vision
vision.plot_rdm('./rdm_vgg', features)
vision.plot_rdm('./rdm_embedding', embedding)

     



