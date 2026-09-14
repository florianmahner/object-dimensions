from objdim.utils import load_sparse_codes, load_image_data
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os
from tqdm import tqdm
import json


def main():
    path_barlow = "data/embeddings/barlowtwins-rn50/avgpool/parameters.npz"
    path_densenet = "data/embeddings/densenet/global_pool/parameters.npz"
    path_clip = "data/embeddings/OpenCLIP/visual/parameters.npz"
    path_resnet = "data/embeddings/resnet50/avgpool/parameters.npz"
    path_vgg = "./data/embeddings/vgg16_bn/classifier.3/parameters.npz"
    path_human = (
        "data/embeddings/human_behavior/parameters.npz"
    )

    paths = [path_barlow, path_densenet, path_clip, path_resnet, path_vgg, path_human]
    names = ["barlow", "densenet", "clip", "resnet", "vgg", "human"]
    filenames = ["model_a", "model_b", "model_c", "model_d", "model_e", "model_f"]
    filenames_to_names = dict(zip(names, filenames))


    mappings = {}

    for path, name in zip(paths, names):
        filter_behavior = True if name == "human" else False

        images, indices = load_image_data(
            "./data/image_data/images12_plus", filter_behavior=filter_behavior
        )

        W = load_sparse_codes(path)



        # Shuffle W column wise and save the mapping of dimension shuffling
        column_shuffle = np.random.permutation(W.shape[1])

        W = W[:, column_shuffle]


        mapping = {key: value.item() for key, value in zip(range(W.shape[1]), column_shuffle)}
        mappings[name] = mapping

        for i, dim in tqdm(enumerate(W.T)):

            top = 9
            fig, axes = plt.subplots(top, top, figsize=(18, 18))

            topk_dim = np.argsort(dim)[::-1][: top * top]
            for j, ax in enumerate(axes.flat):

                val = dim[topk_dim[j]]
                img = io.imread(images[topk_dim[j]])

                if val < 0.5:
                    # make the image gray
                    img = np.ones_like(img) * 122

                ax.imshow(img)
                # if val > 1.0: val = 1.0
                ax.set_title(f"{val:.2f}")
                ax.axis("off")

            fig.tight_layout()
            fig.subplots_adjust(
                left=0, right=1, top=1, bottom=0, wspace=0.015, hspace=0.17
            )

            path = "results/plots/mixing_experiment_anonymous/images/{}_dim_{:02d}.jpg"
            filename = filenames_to_names[name]
            path = path.format(filename, i)
            os.makedirs(os.path.dirname(path), exist_ok=True)

            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)

    # Save the mappings as a json file
    with open("results/plots/mixing_experiment_anonymous/dimension_mapping.json", "w") as f:
        json.dump(mappings, f, indent=4)

    # Save the filenames to names mapping as a json file
    with open("results/plots/mixing_experiment_anonymous/filenames_to_models.json", "w") as f:
        json.dump(filenames_to_names, f, indent=4)


if __name__ == "__main__":
    main()
