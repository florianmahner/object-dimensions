import torch

class PairwiseDataset(torch.utils.data.Dataset):
    ''' Sample pairwise indices from the list combinations'''
    def __init__(self, pairwise_indices, n_samples=None):
        super().__init__()
        self.pairwise_indices = pairwise_indices
        self.n_indices = n_samples if n_samples else len(pairwise_indices) 

    def __getitem__(self, idx):
        return self.pairwise_indices[idx]

    def __len__(self):
        return self.n_indices


class TripletDataset(torch.utils.data.Dataset):
    ''' Sample triplet indices from the list combinations'''
    def __init__(self, triplet_indices, n_samples=None, device=torch.device("cpu")):
        super().__init__()
        self.triplet_indices = triplet_indices
        self.triplets_indices = self.triplet_indices.astype(int)
        # self.triplet_indices = torch.as_tensor(self.triplet_indices)
        # self.triplet_indices = self.triplet_indices.to(device)
        self.n_indices = n_samples if n_samples else len(triplet_indices) 

    def __getitem__(self, idx):
        return self.triplet_indices[idx]

    def __len__(self):
        return self.n_indices

