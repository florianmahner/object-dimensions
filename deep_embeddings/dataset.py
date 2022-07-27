import torch

class PairwiseDataset(torch.utils.data.Dataset):
    ''' Sample pairwise indices from the list combinations'''
    def __init__(self, similarity_mat, n_objects, n_samples=None):
        super().__init__()
        self.n_objects = n_objects
        self.n_samples = n_samples
        self._build_indices()

    def _build_indices(self):
        pairwise_indices = torch.combinations(torch.arange(self.n_objects), 2, with_replacement=False)
        shuffle_indices = torch.randperm(len(pairwise_indices))
        pairwise_indices = pairwise_indices[shuffle_indices]
        pairwise_indices = pairwise_indices[:self.n_samples]
        self.pairwise_indices = pairwise_indices
        self.n_indices = len(pairwise_indices)

    def __getitem__(self, idx):
        indices = self.pairwise_indices[idx]
        
        return indices

    def __len__(self):
        return self.n_indices



