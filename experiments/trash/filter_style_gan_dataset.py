from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


class ImageLoader:
    def __init__(self, image_dir, batch_size):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.filenames = sorted(os.listdir(image_dir))
        self.num_images = len(self.filenames)
        self.current_index = 0

    def __len__(self):
        return (self.num_images + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= self.num_images:
            raise StopIteration
        batch_filenames = self.filenames[
            self.current_index : self.current_index + self.batch_size
        ]
        batch_images = []
        for filename in batch_filenames:
            image_path = os.path.join(self.image_dir, filename)
            with Image.open(image_path) as image:
                image = np.array(image)
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=-1)
                if image.shape[2] == 1:
                    image = np.tile(image, (1, 1, 3))
                batch_images.append(image)
        batch_images = np.array(batch_images)
        self.current_index += self.batch_size
        return batch_images


image_dir = "path/to/images"
batch_size = 1000
image_loader = ImageLoader(image_dir, batch_size)


# Flatten the images into a 2D array
X = images.reshape(images.shape[0], -1)

# Apply PCA to reduce the dimensionality of the images
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X)

# Compute the mini-batch K-means clustering on a subset of the data
n_clusters = topk
batch_size = 10000
kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=42)
kmeans.partial_fit(X_pca)

# Perform the rest of the K-means clustering on the remaining data
for i in range(1, len(X_pca) // batch_size):
    start = i * batch_size
    end = min((i + 1) * batch_size, len(X_pca))
    kmeans.partial_fit(X_pca[start:end])


# Select one representative image from each cluster in parallel
def select_representative(cluster_indices):
    cluster_X = X[cluster_indices]
    distances = np.linalg.norm(cluster_X - np.mean(cluster_X, axis=0), axis=1)
    representative_index = cluster_indices[np.argmax(distances)]
    return representative_index


with Parallel(n_jobs=-1, backend="multiprocessing") as parallel:
    cluster_indices = [np.where(kmeans.labels_ == i)[0] for i in range(n_clusters)]
    representative_indices = parallel(
        delayed(select_representative)(cluster_index)
        for cluster_index in cluster_indices
    )

# Save the representative images to disk
for i, index in enumerate(representative_indices):
    img = images[index]
    img = img.permute(1, 2, 0).numpy()
    img = img_to_uint8(img)
    fig_im, ax_im = plt.subplots(1)
    ax_im.axis("off")
    ax_im.imshow(img)
    for ext in ["png", "pdf"]:
        fname = os.path.join(out_path, f"sampled_image_{i:02d}.{ext}")
        fig_im.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig_im)

    ax = axes[i // 5, i % 5]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    ax.imshow(img)

out_path = os.path.dirname(out_path)
for ext in ["png", "pdf"]:
    fname = os.path.join(out_path, f"{dim}_stylegan.{ext}")
    fig.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0)
plt.close(fig)
