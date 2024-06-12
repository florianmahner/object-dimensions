import numpy as np
import pandas as pd
from scipy.io import loadmat

from objdim.utils import (
    split_half_reliability,
    correlation_matrix,
    rsm_pred_torch,
    correlate_rsms,
    load_concepts,
)


def filter_concepts_by_cls_names(concepts, cls_names):
    """Filter by cls names"""
    cls_names = cls_names[(cls_names != "camera") & (cls_names != "file")]
    cls_names = np.append(cls_names, ["camera1", "file1"])
    # Replace white space with underscore in cls_names
    cls_names = [s.replace(" ", "_") for s in cls_names]
    regex_pattern = "|".join([rf"\b{s}\b" for s in cls_names])
    filtered_concepts = concepts[
        concepts["uniqueID"].str.contains(regex_pattern, regex=True)
    ]
    indices = filtered_concepts.index
    return indices


def main(
    human_embedding,
    dnn_embedding,
    concept_path,
    words48_path,
    human_rdm_gt,
    features,
):

    concept_path = "./data/misc/things_concepts.tsv"
    concepts = load_concepts(concept_path)
    words48 = pd.read_csv(words48_path, sep="\t")
    cls_names = words48["Word"].values
    indices_48 = filter_concepts_by_cls_names(concepts, cls_names)

    rdm_human_true = loadmat(human_rdm_gt)["RDM48_triplet"]
    rsm_human_true = 1 - rdm_human_true

    human_embedding = human_embedding[indices_48]
    rsm_human_pred = rsm_pred_torch(human_embedding)

    reliability = split_half_reliability(rsm_human_true)
    print(f"Reliability of human RSM: {reliability}")
    # TODO still have to spearman brown correct
    image_similarity_human = correlate_rsms(rsm_human_true, rsm_human_pred)
    print(f"Image similarity of human RSM: {image_similarity_human}")

    rsm_dnn_true = correlation_matrix(features[indices_48])
    dnn_embedding = dnn_embedding[indices_48]
    rsm_dnn_pred = rsm_pred_torch(dnn_embedding)
    image_similarity_dnn = correlate_rsms(rsm_dnn_true, rsm_dnn_pred)

    print(f"Image similarity of DNN RSM: {image_similarity_dnn}")
