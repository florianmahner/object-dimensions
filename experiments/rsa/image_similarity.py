import numpy as np
import pandas as pd
from scipy.io import loadmat
import tomlparse

from objdim.utils import (
    split_half_reliability,
    correlation_matrix,
    rsm_pred_torch,
    correlate_rsms,
    load_concepts,
    load_sparse_codes,
    rsm_pred_numpy,
)


def parse_args():
    parser = tomlparse.argparse.ArgumentParser()
    parser.add_argument("--human_embedding_path", type=str)
    parser.add_argument("--dnn_embedding_path", type=str)
    parser.add_argument("--concept_path", type=str)
    parser.add_argument("--words48_path", type=str)
    parser.add_argument("--human_rdm_gt", type=str)
    parser.add_argument("--features", type=str)
    return parser.parse_args()


def filter_concepts_by_cls_names(concepts, cls_names):
    """Filter by cls names. There is something wrong with the order of the concepts and the indices"""
    # Create a copy of cls_names to preserve the original order
    modified_cls_names = cls_names.copy()

    # Remove "camera" and "file" from the copy
    modified_cls_names = modified_cls_names[
        (modified_cls_names != "camera") & (modified_cls_names != "file")
    ]

    # Append "camera1" and "file1" to the end of the copy
    modified_cls_names = np.append(modified_cls_names, ["camera1", "file1"])

    # Replace white space with underscore in modified_cls_names
    modified_cls_names = [s.replace(" ", "_") for s in modified_cls_names]

    # Create a dictionary to map modified_cls_names to their indices
    name_to_index = {name: index for index, name in enumerate(modified_cls_names)}

    # Filter concepts and sort them based on the order in modified_cls_names
    filtered_concepts = concepts[concepts["uniqueID"].isin(modified_cls_names)]
    filtered_concepts = filtered_concepts.sort_values(
        by="uniqueID", key=lambda x: x.map(name_to_index)
    )

    indices = filtered_concepts.index
    return indices


def reconstruct_rsm_48(embedding, indices_48):

    embedding_dot = np.dot(embedding, embedding.T)

    # Exponentiate the dot product matrix
    esim = np.exp(embedding_dot)

    # Initialize cp matrix
    cp = np.zeros((1854, 1854))

    # Main loop to compute cp
    for i in range(1854):
        for j in range(i + 1, 1854):
            ctmp = np.zeros(1854)
            for k in indices_48:
                if k == i or k == j:
                    continue
                ctmp[k] = esim[i, j] / (esim[i, j] + esim[i, k] + esim[j, k])
            cp[i, j] = np.sum(ctmp)

    # Final processing
    cp = cp / len(indices_48)  # Mean calculation
    cp = cp + cp.T  # Make symmetric
    np.fill_diagonal(cp, 1)  # Set diagonal elements to 1

    # Extract vice_sim48 matrix
    sim48 = cp[np.ix_(indices_48, indices_48)]
    return sim48


def main(
    human_embedding_path,
    dnn_embedding_path,
    concept_path,
    words48_path,
    human_rdm_gt,
    features,
):

    human_embedding = load_sparse_codes(human_embedding_path)
    dnn_embedding = load_sparse_codes(dnn_embedding_path)
    concepts = load_concepts(concept_path)

    words48 = pd.read_csv(words48_path, sep=",")
    cls_names = words48["Word"].values
    indices_48 = filter_concepts_by_cls_names(concepts, cls_names)

    rdm_human_true = loadmat(human_rdm_gt)["RDM48_triplet"]
    rsm_human_true = 1 - rdm_human_true

    sim48_human_recon = reconstruct_rsm_48(human_embedding, indices_48)

    # human_embedding = human_embedding[indices_48]
    # rsm_human_pred = rsm_pred_numpy(human_embedding)

    # reliability = split_half_reliability(rsm_human_true)
    # print(f"Reliability of hu÷man RSM: {reliability}")
    # TODO still have to spearman brown correct
    image_similarity_human = correlate_rsms(rsm_human_true, sim48_human_recon)
    print(f"Image similarity of human RSM: {image_similarity_human}")

    rsm_dnn_true = correlation_matrix(features[indices_48])
    dnn_embedding = dnn_embedding[indices_48]
    rsm_dnn_pred = rsm_pred_torch(dnn_embedding)
    image_similarity_dnn = correlate_rsms(rsm_dnn_true, rsm_dnn_pred)

    print(f"Image similarity of DNN RSM: {image_similarity_dnn}")


if __name__ == "__main__":
    args = parse_args()
    main(
        args.human_embedding_path,
        args.dnn_embedding_path,
        args.concept_path,
        args.words48_path,
        args.human_rdm_gt,
        args.features,
    )
