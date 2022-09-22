from thingsvision.utils.data.dataset import ImageDataset
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import matplotlib.gridspec as gridspec

# TODO add image path for load data
# NOTE large things dataset, eacn image per category has suffix _b. -> need to make behavior dataset for visualisation!
# TODO Remove relative paths through smarter way!

parser = argparse.ArgumentParser(description='Visualize embedding')
parser.add_argument('--w_dir', type=str, default='./weights', help='path to weights directory') # TODO remove default
parser.add_argument('--model', type=str, default='vice', help='model name')
parser.add_argument('--n_images', type=int, default=12, choices=[6,12], help='number of images per category')
parser.add_argument('--sort', type=bool, default=True, help='if to sort the dimension row wise')
parser.add_argument('--dir', type=str, default='./plots', help='plotting directory to save the figure')
parser.add_argument('--behavior', type=bool, default=False, help='if behavior images or not')


def load_data(n_images, behavior=False):
    if behavior:
        dataset = ImageDataset(root=f'../reference_images', out_path='', backend='pt')
        idx2obj = None # NOTE maybe for behavior these cannot be extracted!
        obj2idx = None

    else:
        dataset = ImageDataset(root=f'../../../THINGS/image_data/images{n_images}', out_path='', backend='pt')
        idx2obj = dataset.idx_to_cls
        obj2idx = dataset.cls_to_idx

    images = dataset.images

    return idx2obj, obj2idx, images



def plot_per_dim(args):
    assert args.model in ['spose', 'vice']

    path = f"./images_per_dim_{args.model}"
    if not os.path.exists(path):
        os.makedirs(path)

    W = np.loadtxt(args.w_dir)
    _, _, images = load_data(args.n_images, args.behavior)

    print('Shape of weight Matrix', W.shape)

    # sort the dimensions row wise given the positive sum!
    if args.sort:
        W = np.maximum(W, 0) # i kind of have to do this, since negative things are meaningless!
        W = W[:, np.argsort(-np.linalg.norm(W, axis=0, ord=1))]
    
    W  = W.T

    top_k = 16
    top_j = 60 # NOTE this is the number of dimensions to plot

    for dim, w_j in enumerate(W):
        fig = plt.figure(figsize=(4,4))
        gs1 = gridspec.GridSpec(4,4)
        gs1.update(wspace=0.002, hspace=0.002) # set the spacing between axes.
        top_k_samples = np.argsort(-w_j)[:top_k] # this is over the image dimension
        for k, sample in enumerate(top_k_samples):
            ax = plt.subplot(gs1[k])
            ax.imshow(io.imread(images[sample]))
            ax.set_xticks([])
            ax.set_yticks([])
    
        # fig.suptitle("Dimension: {}".format(dim))
        fname = os.path.join(path, f'dim_{dim}.png')
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f'Done plotting for dim {dim}')

        if dim > top_j:
            break    

def plot(args, fname):
    assert args.model in ['spose', 'vice']

    W = np.loadtxt(args.w_dir)
    _, _, images = load_data(args.n_images, args.behavior)


    top_j = 39
    top_k = 12

    n_rows = top_j if top_j <= W.shape[1] else W.shape[1]
    n_cols = top_k
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 60))

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
            # axes[j, k].set_xlabel(f'Weight: {w_j[sample]:.2f}', fontsize=20)
            axes[j, k].set_xticks([])
            axes[j, k].set_yticks([])

        if j == n_rows-1:
            break

    plt.tight_layout()
    fig.savefig(fname, dpi=80)


if __name__ == '__main__':
    args = parser.parse_args()

    # args.w_dir = "../../learned_embeddings/weights_vgg_12_8196bs_adaptive/params/pruned_q_mu_epoch_400.txt"
    args.w_dir = "../../learned_embeddings/weights_vgg_12_512bs/params/pruned_q_mu_epoch_310.txt"
    
    # args.w_dir = "/home/florian/DeepEmbeddings/weights_things_behavior_256bs/params/pruned_q_mu_epoch_500.txt"
    # args.w_dir = "/home/florian/DeepEmbeddings/learned_embeddings/weights_things_behavior_8196bs_adaptive_half/params/pruned_q_mu_epoch_5000.txt"
    args.behavior = False
    args.model = 'vice'
    fname= './all_dimensions/dimensions_vice_512bs_300.png'

    plot_per_dim(args)
    # plot(args, fname=fname)