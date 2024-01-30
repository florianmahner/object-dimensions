class TensorDataloader:
    """Dataloader for tensors when entire dataset is put on gpu and fits into memory"""

    def __init__(self, data, batch_size, shuffle=True, random_seed=42, device="cpu"):
        self.dataset = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.n_samples = data.shape[0]
        self._get_batches()
        self.dataset = torch.from_numpy(self.dataset).to(device)
        self.dataset = self.dataset.type("torch.LongTensor")

    def _get_batches(self):
        # Check if number of samples can be divided by batch size and discard the last batch
        if self.n_samples % self.batch_size != 0:
            print(
                "Number of samples is not divisible by batch size. Discarding last batch."
            )
            self.n_samples = self.n_samples - (self.n_samples % self.batch_size)
            self.dataset = self.dataset[: self.n_samples]
            self.n_batches = self.n_samples // self.batch_size

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.shuffle:
            np.random.seed(self.random_seed)
            indices = np.random.permutation(self.n_samples)
        else:
            indices = np.arange(self.n_samples)

        for i in range(self.n_batches):
            batch_indices = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield self.dataset[batch_indices]
