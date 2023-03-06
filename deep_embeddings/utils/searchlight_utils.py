import os
import torch
import glob
import pickle

# import cv2
import matplotlib.pyplot as plt
import numpy as np


def zero_padding(img, window_size):
    eps = img.min()
    h_pad = np.ones((img.shape[0], window_size // 2)) * eps
    img = np.hstack((h_pad, img, h_pad))
    v_pad = np.ones((window_size // 2, img.shape[1])) * eps
    img = np.vstack((v_pad, img, v_pad))

    return img


def save_masked_image(mask, img):
    # Add a mask to the image as a heatmap overlayed
    def add_mask(img, mask, alpha=0.5):
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        mask = np.float32(mask) / 255
        img = img / 255
        print(img.max(), mask.max())
        img = img * (1 - alpha) + mask * alpha
        img = np.clip(img, 0, 1)
        return img

    img = add_mask(img, mask)
    plt.figure()
    plt.imshow(img)


# mask = (mask - np.min(mask)) / np.max(mask)
# mask = 1 - mask

# plt.figure()
# plt.imshow(img)
# plt.imshow((mask * 255).astype('uint8'), cmap='viridis', alpha=1)
# plt.savefig('heatmap.png')


def rescale_uint8(img):
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)

    return img


def load_searchlight_imgs(path):
    # NOTE make use of glob here instead!
    s_path = os.path.join(path, "**", "*.pkl")
    searchlight_results = sorted(glob.glob(s_path, recursive=True))

    deltas = []
    imgs = []
    for path in searchlight_results:

        print(path)
        with open(path, "rb") as f:

            save_dict = pickle.load(f)
            print(save_dict.keys())
            delta = save_dict["diffs"]
            img = save_dict["img"].squeeze()
            img = img.T
            img = rescale_uint8(img)
            imgs.append(img)
            window_size = img.shape[0] - delta.shape[0]

            print(img.shape, delta.shape, window_size)
            delta = zero_padding(delta, window_size)
            deltas.append(delta)

    deltas = np.asarray(deltas)
    imgs = np.asarray(imgs)
    return deltas, imgs, searchlight_results


def load_features(activation_path):
    with open(os.path.join(activation_path, "features.npy"), "rb") as f:
        features = np.load(f)
    return features


def get_file_names(activation_path):
    return (
        open(os.path.join(activation_path, "file_names.txt"), "r").read().splitlines()
    )


def get_activation(activations, name):
    def hook(model, input, output):
        activations[name] = output.detach()

    return hook


def register_hook(model):
    for name, module in model.named_modules():
        module.register_forward_hook(get_activation(name))
    return model


def get_x_window(H, y_pos, window_size):
    x_top_diff = abs(0 - y_pos)
    mask_x_start = (
        y_pos - x_top_diff if (x_top_diff < window_size) else y_pos - window_size
    )
    x_bottom_diff = abs(H - y_pos)
    mask_x_end = (
        (y_pos + x_bottom_diff + 1)
        if (x_bottom_diff < window_size)
        else (y_pos + window_size + 1)
    )

    return mask_x_start, mask_x_end


def get_y_window(W, x_pos, window_size):
    y_top_diff = abs(0 - x_pos)
    mask_y_start = (
        x_pos - y_top_diff if (y_top_diff < window_size) else x_pos - window_size
    )
    y_bottom_diff = abs(W - x_pos)
    mask_y_end = (
        (x_pos + y_bottom_diff + 1)
        if (y_bottom_diff < window_size)
        else x_pos + window_size + 1
    )
    return mask_y_start, mask_y_end


def mask_img(img, y_pos, x_pos, window_size):
    _, _, H, W = img.shape  # img shape = [B x C x H x W]
    mask_x_start, mask_x_end = get_x_window(H, y_pos, window_size)
    mask_y_start, mask_y_end = get_y_window(W, x_pos, window_size)
    img[
        :, :, mask_x_start:mask_x_end, mask_y_start:mask_y_end
    ] = 0  # .5 = grey mask; 0. = black mask

    return img


def get_latents(latent_path, latent_dim, latent_version, device):
    subfolder = (
        "sampled_latents" if latent_version == "sampled" else "optimized_latents"
    )
    latent_path = os.path.join(latent_path, f"{latent_dim:02d}", subfolder)
    sampled_latents = []
    for f in sorted(os.listdir(latent_path)):
        if f.endswith("pt"):
            sampled_latents.append(
                torch.load(os.path.join(latent_path, f), map_location=device)
                .cpu()
                .numpy()
                .tolist()
            )
    return torch.tensor(sampled_latents)


def get_codes_and_images(generator, comparator, latents, truncation, device):
    generator.to(device)
    comparator.to(device)
    generator.eval()
    comparator.eval()
    with torch.no_grad():
        latents = latents.to(device)
        images = generator(latents, truncation)
        codes, _ = comparator(images)
    codes = codes.cpu().numpy()
    return codes, images
