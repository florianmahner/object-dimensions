from vi_embeddings.model import VI
from vi_embeddings.priors import SpikeSlabPrior, ExponentialPrior
from vi_embeddings.dataset import PairwiseDataset
from vi_embeddings.utils import build_similarity_mat
from vi_embeddings.engine import MLTrainer
from vi_embeddings.loggers import DefaultLogger, FileLogger, ParameterLogger, TensorboardLogger
from config import Config as cfg
from sklearn.model_selection import train_test_split

import torch
import numpy as np
import os


def train():
    device = torch.device('cuda:0')
    torch.manual_seed(cfg.rnd_seed)
    np.random.seed(cfg.rnd_seed)

    features = np.load(cfg.feature_path)
    n_objects = features.shape[0]

    similarity_mat = build_similarity_mat(features, init_dim=cfg.init_dim)

    model = VI(n_objects, cfg.init_dim)
    model.to(device)

    if cfg.prior == 'sslab':
        prior = SpikeSlabPrior(n_objects, cfg.init_dim, spike=0.25, slab=1.0, pi=0.5)
    elif cfg.prior == 'exp':
        prior = ExponentialPrior(n_objects, lmbda=1.0)


    indices = torch.combinations(torch.arange(n_objects), 2, with_replacement=False)
    rand_ind = torch.randperm(len(indices))
    indices = indices[rand_ind]
    train_indices, val_indices = train_test_split(indices, test_size=cfg.val_split)

    train_dataset = PairwiseDataset(train_indices)
    val_dataset = PairwiseDataset(val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=False)

    # Build loggers and 
    logger = DefaultLogger(cfg.log_path)
    logger.add_logger('params', ParameterLogger(cfg.log_path, model, ['sorted_pruned_params']), update_interval=cfg.params_interval)
    if cfg.tensorboard:
        logger.add_logger('tensorboard', TensorboardLogger(cfg.log_path), callbacks=['train_loss', 'val_loss'], update_interval=1)

    logger.add_logger('file', FileLogger(cfg.log_path), update_interval=1)

    trainer = MLTrainer(model, prior, train_loader, val_loader, logger, similarity_mat, device)
    trainer.parse_from_config(cfg)

    if cfg.load_model:
        model_path = os.path.join(cfg.log_path, cfg.model_name)
        trainer.init_model_from_checkpoint(model_path)

    trainer.train()


if __name__ == '__main__':
    train()
