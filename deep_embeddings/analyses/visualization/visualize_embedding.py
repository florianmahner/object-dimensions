from thingsvision.dataset import ImageDataset
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
import argparse

# TODO add image path for load data
# NOTE large things dataset, eacn image per category has suffix _b. -> need to make behavior dataset for visualisation!

parser = argparse.ArgumentParser(description='Visualize embedding')
parser.add_argument('--w_dir', type=str, default='./weights', help='path to weights directory') # TODO remove default
parser.add_argument('--model', type=str, default='vice', help='model name')
parser.add_argument('--n_images', type=int, default=12, choices=[6,12], help='number of images per category')
parser.add_argument('--sort', type=bool, default=True, help='if to sort the dimension row wise')
parser.add_argument('--dir', type=str, default='./plots', help='plotting directory to save the figure')
parser.add_argument('--add_ref_imgs', type=bool, default=True, help='add reference images')


def load_data(n_images, add_ref_imgs):
    dataset = ImageDataset(root=f'../../THINGS/image_data/images{n_images}', out_path='', backend='pt', 
                                        imagenet_train=False, imagenet_val=False,
                                        things=True, things_behavior=True, add_ref_imgs=add_ref_imgs,
                                        transforms=None)


    idx2obj = dataset.idx_to_cls
    obj2idx = dataset.cls_to_idx
    images = dataset.images

    breakpoint()

    return idx2obj, obj2idx, images
    

def plot(args):
    assert args.model in ['spose', 'vice']

    W = np.loadtxt(args.w_dir)
    _, _, images = load_data(args.n_images, args.add_ref_imgs)

    top_j = 39
    top_k = 12

    n_rows = top_j if top_j <= W.shape[0] else W.shape[0]
    n_cols = top_k
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(40, 120))

    print('Shape of weight Matrix', W.shape)

    # sort the dimensions row wise given the positive sum!
    if args.sort:
        W = np.maximum(W, 0) # i kind of have to do this, since negative things are meaningless!
        W = W[:, np.argsort(-np.linalg.norm(W, axis=0, ord=1))]
        W  = W.T

    for j, w_j in enumerate(W):
        top_k_samples = np.argsort(-w_j)[:top_k] # this is over the image dimension
        for k, sample in enumerate(top_k_samples):
            axes[j, k].imshow(io.imread(images[sample]))
            axes[j, k].set_xlabel(f'Weight: {w_j[sample]:.2f}', fontsize=20)
            axes[j, k].set_xticks([])
            axes[j, k].set_yticks([])

        if j == n_rows-1:
            break

    plt.tight_layout()
    fig.savefig(f'dimensions_{args.model}.png', dpi=50)


if __name__ == '__main__':
    args = parser.parse_args()

    # args.w_dir = '/LOCAL/fmahner/DeepEmbeddings/weights_sslab12_20mio_cosine_gamma_097/params/pruned_q_var_epoch_200.txt'
    # args.w_dir = "../weights_triplets_50mio/params/pruned_q_mu_epoch_500.txt"
    args.w_dir = "/home/florian/DeepEmbeddings/weights_things_behavior_256bs/params/pruned_q_mu_epoch_500.txt"
    plot(args)