import torch
import math
import os
import random

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

class ElasticNet(nn.Module):
    """ Elastic Net regression with L1 and L2 penalty on the weights. """

    def __init__(self, in_features, out_features, alpha=1.0, l1_ratio=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._alpha = alpha
        self._l1_ratio = l1_ratio
        
        # We want to learn the coef (weights) and intercept (bias)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    @property 
    def alpha(self):
        return self._alpha

    @property 
    def l1_ratio(self):
        return self._l1_ratio

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def l1_penalty(self):
        return self.alpha * self._l1_ratio * torch.norm(self.weight, 1)

    def l2_penalty(self):
        return 0.5 * self._alpha * (1 - self._l1_ratio) * torch.norm(self.weight, 2)

    def elastic_penalty(self):
        return self.l1_penalty() + self.l2_penalty()

    def get_loss(self, x, y):
        """ Combined loss function for Elastic Net with L1 and L2 penalty. """
        return F.mse_loss(self(x), y) + self.elastic_penalty()


def load_activations(path):
    if path.endswith('npy'):
        with open(path, 'rb') as f:
            act = np.load(f)
    else:
        act = np.loadtxt(path)
    return act

def center_activations(act):
    return act - act.mean(axis=0)


def remove_zeros(W, eps=.1):
    w_max = np.max(W, axis=1)
    W = W[np.where(w_max > eps)]
    return W

def get_weights(path):
    W = np.loadtxt(os.path.join(path))
    return remove_zeros(W)

def load_sparse_codes(path):
    W = get_weights(path)
    l1_norms = np.linalg.norm(W, ord=1, axis=1)
    sorted_dims = np.argsort(l1_norms)[::-1]
    W = W[sorted_dims]
    return W, sorted_dims


def train_test_split(X, y, train_frac=.9, shuffle=True):
    """split the data into train and test sets"""
    N = X.shape[0]
    if shuffle:
        rnd_perm = np.random.permutation(N)
        train_indices = rnd_perm[:int(N*train_frac)]
        test_indices = rnd_perm[int(N*train_frac):]
    else:
        train_indices = np.arange(0, int(N*train_frac))
        test_indices = np.arange(int(N*train_frac), N)
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


def save_regression_predictions(weight, bias, path):
    coef = weight.cpu().numpy()
    intercept = bias.cpu().numpy()
    out_path = os.path.join(path, 'regression_predictions.npy')
    np.savez_compressed(out_path, intercept=intercept, coef=coef)    

def train():
    rnd_seed = 42

    np.random.seed(rnd_seed)
    random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)

    k_folds = 5
    lmbda = 1.0
    max_iter = 5000
    dnn_path = '/home/florian/THINGS/vgg_bn_features_behavior/features.npy'
    embedding_path = "../learned_embeddings/weights_things_behavior_8196bs_adaptive_half/params/pruned_q_mu_epoch_5000.txt"

    out_path = './sparse_code_predictions'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # TODO outsource positivity constraint!
    act = load_activations(dnn_path)
    X = center_activations(act)
    X = act


    X = torch.from_numpy(X).float().to(device)
    X = F.relu(X)

    y, _ = load_sparse_codes(embedding_path)

    y = torch.from_numpy(y).float().to(device)
    y = F.relu(y)

    y = y[:,0].unsqueeze(1) # take only one dimension of the embedding for debugging testing purpose

    # I have to separately learn a regression model for each dimension! how can I do this efficiently?
    model = ElasticNet(in_features=X.shape[1], out_features=y.shape[1], alpha=1.0, l1_ratio=lmbda)
    model.to(device)

    # TODO need to also cross validate across differnet lambdas!!

    kf = KFold(n_splits=k_folds, random_state=rnd_seed, shuffle=True)
    r2_scores = np.zeros(k_folds)    
    for k, (train_idx, test_idx) in enumerate(kf.split(X)):
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        r2 = np.inf # stores the best r2 score for current fold evaluated on val set
        best_val_loss = np.inf

        best_weights = np.inf
        best_bias = np.inf
        best_iter = np.inf

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        for iter in range(max_iter):
            
            optimizer.zero_grad()
            train_loss = model.get_loss(X_train, y_train)
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                y_pred = model(X_test)
                val_loss = model.get_loss(X_test, y_test)
                
                if iter % 100 == 0:
                    print(f"Epoch: {iter}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}")
                    # print(model.weight.max())

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_iter = iter

                    r2 = r2_score(y_test.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
                    best_weights = model.weight.detach()
                    best_bias = model.bias.detach()
    

        # breakpoint()

        print(f"Best Val Loss for split {k+1} at iter {best_iter}: {best_val_loss}, R2 Score: {r2}\n")
        r2_scores[k] = r2 # r2 score for that train val split!
        save_regression_predictions(best_weights, best_bias, out_path) # TODO check when exactly to store the coefficients and intercepts (i.e. what is the best model?)

    
    print(f"Mean R2 Score over splits: {np.mean(r2_scores)}")



if __name__ == '__main__':
    train()