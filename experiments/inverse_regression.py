import numpy as np
from deep_embeddings.utils.utils import load_sparse_codes, load_deepnet_activations
from deep_embeddings.utils.latent_predictor import load_regression_weights
from sklearn.metrics import r2_score
from deep_embeddings.utils.utils import correlate_rsms


def invert_regression(B, X, y, dtype=np.float64):
    B = B.astype(dtype)
    X = X.astype(dtype)
    y = y.astype(dtype)

    # We estimate the covariance as the expected value of the squared error of the
    # residuals (i.e. the error of the predictions) -> need to do this on held out data!
    y_hat = (X @ B).T
    y_hat = np.maximum(y_hat, 0)  # ReLU
    residuals = y - y_hat

    # Compute the MSE between the predictions and the true values
    mse = np.mean(residuals**2, axis=1)

    # Compute the covariance matrix of the errors
    # cov_y = np.diag(mse) + 1e-3
    # Adding the coefficients makes it worse
    cov_y = np.eye(y.shape[0]) * 1e-3

    # Covariance matrix of activations, (4096, 4096), (i.e. the PRIOR)
    # Either estimated from data (held out acts) or identity matrix
    # cov_x = X.T @ X / (X.shape[0] - 1)
    cov_x = np.eye(X.shape[1]) + np.eye(X.shape[1]) * 1e-6

    # mean of the posterior of the betas
    m_posterior = np.linalg.inv(B @ np.linalg.inv(cov_y) @ B.T + cov_x)  # (4096, 4096)
    m_posterior = m_posterior @ B @ np.linalg.inv(cov_y) @ y

    # we just take the MAP estimate of the posterior as our inverted regression coefficients!
    X_inv = m_posterior.T

    return X_inv


X = load_deepnet_activations(
    "./data/triplets/vgg16_bn/classifier.3/",
    zscore=False,
    center=False,
    relu=True,
    to_torch=False,
)
y = load_sparse_codes(
    "./results/sslab_final/deep/vgg16_bn/classifier.3/20.mio/sslab/300/256/0.24/0/params/params_epoch_2000.npz",
    zscore=False,
)
B, _ = load_regression_weights(
    "./results/sslab_final/deep/vgg16_bn/classifier.3/20.mio/sslab/300/256/0.24/0/analyses/sparse_codes",
    X.shape[1],
    y.shape[1],
    to_numpy=True,
)
y, B = y.T, B.T

map_X = invert_regression(B, X, y)

X_rsm = X @ X.T
X_map_rsm = map_X @ map_X.T
rho = correlate_rsms(X_rsm, X_map_rsm)
print("RSM MAP r", rho)

B_pinv = np.linalg.pinv(B)
map_X_pinv = y.T @ B_pinv
rho = correlate_rsms(X_rsm, map_X_pinv @ map_X_pinv.T)
print("RSM PInv r", rho)
