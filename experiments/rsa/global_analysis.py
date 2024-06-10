import os
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from object_dimensions.utils import correlate_rsms, rsm_pred_torch


def filter_rsm_by_concepts(
    rsm_human: np.ndarray, rsm_dnn: np.ndarray, concepts: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Sort human and DNN by their assigned concept categories, so that
    objects belonging to the same concept are grouped together in the RSM"""

    def get_singletons(concepts):
        set_union = concepts.sum(axis=1)
        unique_memberships = np.where(set_union > 1.0, 0.0, set_union).astype(bool)
        singletons = concepts.iloc[unique_memberships, :]
        non_singletons = concepts.iloc[~unique_memberships, :]
        return singletons, non_singletons

    def sort_singletons(singletons):
        return np.hstack(
            [
                singletons[singletons.loc[:, concept] == 1.0].index
                for concept in singletons.keys()
            ]
        )

    singletons, non_singletons = get_singletons(concepts)
    singletons = sort_singletons(singletons)
    non_singletons = np.random.permutation(non_singletons.index)
    sorted_items = np.hstack((singletons, non_singletons))

    rsm_human = rsm_human[sorted_items, :]
    rsm_human = rsm_human[:, sorted_items]

    rsm_dnn = rsm_dnn[sorted_items, :]
    rsm_dnn = rsm_dnn[:, sorted_items]
    rsm_human = rankdata(rsm_human).reshape(rsm_human.shape)
    rsm_dnn = rankdata(rsm_dnn).reshape(rsm_dnn.shape)
    return rsm_human, rsm_dnn


def global_rsa_analysis(human_embedding, dnn_embedding, concepts: pd.DataFrame):
    """Performs RSA between the human and DNN embeddings based on triplet reconstructions
    of all entries in the RSM."""
    rsm_human = rsm_pred_torch(human_embedding)
    rsm_dnn = rsm_pred_torch(dnn_embedding)
    corr = correlate_rsms(rsm_human, rsm_dnn, "pearson")
    rsm_human, rsm_dnn = filter_rsm_by_concepts(rsm_human, rsm_dnn, concepts)
    return rsm_human, rsm_dnn, corr


def plot_rsms(human_rsm, dnn_rsm, plot_dir):
    """Plot the RSMs of the human and DNN embeddings."""
    for name, rsm in zip(["human", "dnn"], [human_rsm, dnn_rsm]):
        fname = os.path.join(plot_dir, f"{name}_rsm.jpg")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.axis("off")
        ax.imshow(rsm, cmap="viridis", interpolation="nearest")
        fig.savefig(fname, pad_inches=0, bbox_inches="tight", dpi=450)
        plt.close(fig)
