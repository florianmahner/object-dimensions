import os
import glob
import numpy as np
import torch
from pathlib import Path
from scipy.stats import pearsonr


def load_deepnet_activations(
    activation_path, center=False, zscore=False, to_torch=False, relu=True
):
    """Load activations from a .npy file"""
    if center and zscore:
        raise ValueError("Cannot center and zscore activations at the same time")
    activation_path = glob.glob(os.path.join(activation_path, "*.npy"), recursive=True)

    if len(activation_path) > 1:
        raise ValueError("More than one .npy file found in the activation path")
    activation_path = activation_path[0]
    if activation_path.endswith("npy"):
        with open(activation_path, "rb") as f:
            act = np.load(f)
    else:
        act = np.loadtxt(activation_path)
    # We also add the positivity constraint here when loading activities!
    act = transform_activations(act, zscore=zscore, center=center, relu=relu)
    if to_torch:
        act = torch.from_numpy(act)

    return act


def transform_activations(act, zscore=False, center=False, relu=False):
    """Transform activations"""
    if center and zscore:
        raise ValueError("Cannot center and zscore activations at the same time")
    if relu:
        act = relu_embedding(act)
    # We standardize or center AFTER the relu. neg vals. are then meaningful
    if center:
        act = center_activations(act)
    if zscore:
        act = zscore_activations(act)

    return act


def center_activations(act):
    return act - act.mean(axis=0)


def zscore_activations(act, dim=0, eps=1e-8):
    std = np.std(act, axis=dim) + eps
    mean = np.mean(act, axis=dim)
    return (act - mean) / std


def relu_embedding(W):
    return np.maximum(0, W)


def create_results_path(embedding_path, *args, base_path="./results"):
    """Create a path if it does not exist. Each argument is a subdirectory in the path."""
    import os

    try:
        model_name = Path(embedding_path).parts[-2]
    except IndexError:
        raise ValueError("Invalid embedding path: unable to extract model name")
    out_path = os.path.join(base_path, "experiments", model_name, *args)

    try:
        os.makedirs(out_path, exist_ok=True)
    except OSError as os:
        raise OSError("Error creating path {}: {}".format(out_path, os))

    return out_path


def remove_zeros(W, eps=0.1):
    w_max = np.max(W, axis=1)
    W = W[np.where(w_max > eps)]
    return W


def transform_params(weights, scale, relu=True):
    """We transform by (i) adding a positivity constraint and the sorting in descending order"""
    if relu:
        weights = relu_embedding(weights)
    sorted_dims = np.argsort(-np.linalg.norm(weights, axis=0, ord=1))

    weights = weights[:, sorted_dims]
    scale = scale[:, sorted_dims]
    d1, d2 = weights.shape
    # We transpose so that the matrix is always of shape (n_images, n_dims)
    if d1 < d2:
        weights = weights.T
        scale = scale.T

    return weights, scale, sorted_dims


def load_sparse_codes(
    path, weights=None, vars=None, with_dim=False, with_var=False, relu=True
):
    """Load sparse codes from a directory. Can either be a txt file or a npy file or a loaded array of shape (n_images, n_dims)"""
    if weights is not None and vars is not None:
        assert isinstance(weights, np.ndarray) and isinstance(
            vars, np.ndarray
        ), "Weights and var must be numpy arrays"

    file = glob.glob(os.path.join(path, "parameters.npz"))
    if len(file) > 0:
        params = np.load(os.path.join(path, "parameters.npz"))
        if params["method"] == "variational":
            weights = params["pruned_q_mu"]
            vars = params["pruned_q_var"]
        else:
            weights = params["pruned_weights"]
            vars = np.zeros_like(weights)
    elif isinstance(path, str):
        if path.endswith(".txt"):
            # Check if q_mu or q_var in path
            if "q_mu" or "q_var" in path:
                try:
                    weights = np.loadtxt(path.replace("q_var", "q_mu"))
                    vars = np.loadtxt(path.replace("q_mu", "q_var"))
                except OSError:
                    raise OSError(
                        "Error loading sparse codes from path {}".format(path)
                    )
            else:
                if "embedding" in os.path.basename(path):
                    weights = np.loadtxt(path)
                    vars = None

        elif path.endswith(".npz"):
            params = np.load(path)
            if params["method"] == "variational":
                weights = params["pruned_q_mu"]
                vars = params["pruned_q_var"]

            else:
                weights = params["pruned_weights"]
                vars = np.zeros_like(weights)

    elif isinstance(path, np.lib.npyio.NpzFile):
        if path["method"] == "variational":
            weights = path["pruned_q_mu"]
            vars = path["pruned_q_var"]
        else:
            weights = path["pruned_weights"]
            vars = np.zeros_like(weights)

    else:
        raise ValueError(
            "Weights or Vars must be a .txt file path or as numpy array or .npz file"
        )

    weights, vars, sorted_dims = transform_params(weights, vars, relu=relu)
    if with_dim:
        if with_var:
            return weights, vars, sorted_dims
        else:
            return weights, sorted_dims
    else:
        if with_var:
            return weights, vars
        else:
            return weights


def pairiwise_correlate_dimensions(
    weights_human,
    weights_dnn,
    base="human",
    duplicates=False,
    sort_by_corrs=True,
    return_corrs=True,
):
    """Correlate the weights of two modalities and return the weights of both modalities in eiter the same or different orders
    Parameters:
    weights_human (np.ndarray): The weights of the human modality
    weights_dnn (np.ndarray): The weights of the DNN modality
    base (str): The modality that will be used as the base for the comparison.
                The other modality will be compared to this one.
    duplicates (bool): Whether to allow duplicate dimensions in the comparison modality.
                       If set to False, we correlate without repeats.
    sort_by_corrs (bool): Whether to sort the dimensions based on the highest correlations. Otherwise we sort
                            the dimensions based on the order of the dimensions in the base modality
                            (i.e. the sum of the weights)
    Returns:
    Tuple[np.ndarray, np.ndarray]: The correlated weights of the human and DNN modalities,
    in either the same or different orders based on the parameters.
    """
    dim_human, dim_dnn = weights_human.shape[1], weights_dnn.shape[1]
    weights = {"human": weights_human, "dnn": weights_dnn}
    dims = {"human": dim_human, "dnn": dim_dnn}
    weights_base = weights[base]
    dim_base = dims[base]
    weights_comp = weights["dnn" if base == "human" else "human"]
    dim_comp = dims["dnn" if base == "human" else "human"]

    if dim_base > dim_comp and not duplicates:
        raise ValueError(
            """If duplicates is set to False, the number of dimensions in the base modality
            must be smaller than the number of dimensions in the comparison modality."""
        )

    matching_dims, matching_corrs = [], []
    for i, w1 in enumerate(weights_base.T):
        corrs = np.zeros(dim_comp)
        for j, w2 in enumerate(weights_comp.T):
            corrs[j] = pearsonr(w1, w2)[0]

        sorted_dim_corrs = np.argsort(-corrs)
        if duplicates:
            matching_dims.append(sorted_dim_corrs[0])
        else:
            for dim in sorted_dim_corrs:
                if (
                    dim not in matching_dims
                ):  # take the highest correlation that has not been used before
                    matching_dims.append(dim)
                    break

        # Store the highest correlation for the selected dimension
        select_dim = matching_dims[-1]
        matching_corrs.append(corrs[select_dim])

    # Now sort the dimensions based on the highest correlations
    if sort_by_corrs:
        matching_corrs = np.array(matching_corrs)
        sorted_corrs = np.argsort(-matching_corrs)
        matching_corrs = matching_corrs[sorted_corrs]
        comp_dims = np.array(matching_dims)[sorted_corrs]
        base_dims = sorted_corrs

    else:
        base_dims = np.arange(len(matching_dims))
        comp_dims = np.array(matching_dims)

    weights_base = weights_base[:, base_dims]
    weights_comp = weights_comp[:, comp_dims]

    if return_corrs:
        return weights_base, weights_comp, matching_corrs, matching_dims

    else:
        return weights_base, weights_comp
