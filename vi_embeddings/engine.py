import numpy as np
from sklearn.covariance import log_likelihood
import torch
import os
from copy import deepcopy
import logging
import torch.nn.functional as F
import vi_embeddings.utils as utils


class MLTrainer:
    def __init__(
        self,
        model,
        prior,
        train_loader,
        val_loader,
        similarity_mat,
        device="cpu",
        log_path="./weights",
        n_epochs=100,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_laoder = val_loader
        self.similarity_mat = similarity_mat
        self.lr = 0.001
        self.mc_samples = 5
        self.n_epochs = n_epochs
        self.log_path = log_path
        self._init_training_params()
        self._build_optimizer()

        self.prior = prior
        self.device = device
        self.model.to(self.device)
        self.prior.to(self.device)

    def _build_path(self):
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

    def _init_training_params(self):
        self.start_epoch = 1
        self.loss = 0
        self.best_loss = np.inf
        self.checkpoint_interval = 50
        self.weight_interval = 20

    def _build_optimizer(self):
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _init_optimizer(self, state_dict):
        self.optim = self.optim.load_state_dict(state_dict)

    def init_model_from_checkpoint(self, checkpoint):
        print("Load model and optimizer from state dict and continue training")
        checkpoint = torch.load(checkpoint)
        saved_args = checkpoint["args"]
        self._init_optimizer(checkpoint["optim_state_dict"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.start_epoch = checkpoint["epoch"]
        self.loss = checkpoint["loss"]
        self.n_epochs = self.start_epoch + self.n_epochs
        self.lr = saved_args.lr

    def save_checkpoint(self, epoch):
        # save model!
        m_path = os.path.join(self.log_path, "checkpoint_epoch_{}.tar")
        torch.save(
            {
                "model_state_dict": deepcopy(self.model.state_dict()),
                "optim_state_dict": deepcopy(self.optim.state_dict()),
                "train_loader": self.train_loader,
                "val_loader": self.val_loader,
                "epoch": epoch,
            },
            m_path.format(epoch),
        )

    def step_batch(self, indices):
        indices = indices.to(self.device)
        indices = indices.unbind(1)  # convert into two lists of x,y indices

        sim_features = self.similarity_mat[indices]
        sim_features = sim_features.to(self.device)

        embedding, loc, scale = self.model()

        sim_embedding = self.compute_embedding_similarities(embedding, indices)
        log_likelihood = F.mse_loss(sim_features, sim_embedding)

        # log probability of variational distribution
        log_q = utils.normalized_pdf(embedding, loc, scale).log()

        # gaussian prior log probability of the prior distribution
        log_p = self.prior(embedding).log()

        return log_likelihood, log_q, log_p

    def step_dataloader(self, dataloader):
        loss_history = []
        n_pairwise = len(dataloader.dataset)
        n_batches = len(dataloader)

        for k, indices in enumerate(dataloader):

            # only for trainin batches, not val batches
            if self.model.training:
                log_likelihood, log_q, log_p = self.step_batch(indices)

                loss = log_likelihood + complexity_loss
                complexity_loss = (1 / n_pairwise) * (log_q.sum() - log_p.sum())

                # faster alternative to optim.zero_grad()
                for param in self.model.parameters():
                    param.grad = None
                loss.backward()
                self.optim.step()
                loss = loss.item()

                logging.info(
                    f"Train Batch {k}/{n_batches} Likelihood {log_likelihood.item()}, Complexity {complexity_loss.item()}",
                    end="\r",
                )

            # we do mc sampling of the variational posterior for validation batches
            else:
                sampled_likelihoods = torch.zeros((self.mc_samples, self.n_dimensions))
                for k in self.mc_samples:
                    log_likelihood, _, _ = self.step_batch(indices)
                    sampled_likelihoods[k] = log_likelihood.item()

                # validation loss
                loss = torch.mean(sampled_likelihoods)

                logging.info(
                    f"Val Batch {k}/{n_batches} Likelihood {loss.item()}",
                    end="\r",
                )

            loss_history.append(loss)

        return loss_history

    def train_one_epoch(self):
        self.model.train(True)
        train_loss = self.step_dataloader(self.train_loader)

        return train_loss

    @torch.no_grad()
    def evaluate_one_epoch(self):
        self.model.eval()
        val_loss = self.step_dataloader(self.val_loader)

        return val_loss

    def train(self):
        self.model.to(self.device)
        for epoch in range(self.start_epoch, self.n_epochs + 1):

            train_loss = self.train_one_epoch()
            val_loss = self.evaluate_one_epoch()

            logging.info(f"Epoch {epoch} - Loss train: {train_loss.mean():>14.6f}")
            logging.info(f"Epoch {epoch} - Loss val: {val_loss.mean():>15.6f}")

            if epoch % self.weight_interval == 0:
                self.store_embeddings(epoch)

            if epoch % self.checkpoint_interval == 0:
                self.save_checkpoint(epoch)

    def prune_dimensions(self):
        signal, pruned_loc, pruned_scale = self.model.prune_dimensions(alpha=0.05)
        return pruned_loc

    def store_embeddings(self, epoch):
        embedding = self.model.detached_params()["q_mu"]
        pruned_embedding = self.prune_dimensions()
        f_path = os.path.join(self.log_path, "{}_epoch_{}.txt")
        np.savetxt(f_path.format("weights", epoch), embedding)
        np.savetxt(f_path.format("weights_pruned", epoch), pruned_embedding)

    def correlate_embeddings(self):
        pass

    def compute_embedding_similarities(self, embedding, indices):
        indices_i, indices_j = indices
        embeddings_i = F.relu(embedding[indices_i])
        embeddings_j = F.relu(embedding[indices_j])
        sim_embedding = torch.sum(
            embeddings_i * embeddings_j, dim=1
        )  # this is the dot produt of the embedding vectors

        return sim_embedding
