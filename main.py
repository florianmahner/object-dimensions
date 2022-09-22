from deep_embeddings.core.model import VI
from deep_embeddings.core.priors import SpikeSlabPrior, ExponentialPrior
from deep_embeddings.core.dataset import TripletDataset
from deep_embeddings.core.engine import MLTrainer
from deep_embeddings.logging.loggers import DeepEmbeddingLogger
from config import Config

import torch
import numpy as np
import os


def build_triplet_dataset(triplet_path, device):
    train = np.load(os.path.join(triplet_path, "train_90.npy"))
    test = np.load(os.path.join(triplet_path, "test_10.npy"))

    # maybe need to do train/val split here beforehand and test=test?
    train_dataset = TripletDataset(train, device=device)
    val_dataset = TripletDataset(test, device=device)

    return train_dataset, val_dataset

def train(cfg):
    assert (cfg.fresh and cfg.load_model) is not True, "You can either load a model (config.load_model) or train a new one (config.fresh), not both!"
    
    torch.manual_seed(cfg.rnd_seed)
    np.random.seed(cfg.rnd_seed)

    features = np.load(cfg.feature_path)
    n_objects = features.shape[0]

    if cfg.modality == "behavior":
        n_objects = 1854

    model = VI(n_objects, cfg.init_dim)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=[0,1,2])
    
    else:
        model = torch.nn.DataParallel(model, device_ids=[0])
    model.to(device)

    if cfg.prior == 'sslab':
        prior = SpikeSlabPrior(n_objects, cfg.init_dim, spike=0.25, slab=1.0, pi=0.5)
        # prior = SpikeSlabPriorConstrained(n_objects, cfg.init_dim, spike=0.25, slab=1.0, pi=0.5)
    elif cfg.prior == 'exp':
        prior = ExponentialPrior(n_objects, lmbda=1.0)
    else:
        raise NameError('Unknown prior')

    train_dataset, val_dataset = build_triplet_dataset(cfg.triplet_path, device)

    # If all data on GPU, num workers need to be 0 and pin memory false
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Build loggers and 
    logger = DeepEmbeddingLogger(model, cfg)

    trainer = MLTrainer(model, prior, train_loader, val_loader, logger, device)
    trainer.parse_from_config(cfg)

    trainer.train()


if __name__ == '__main__':
    cfg = Config()
    train(cfg)
