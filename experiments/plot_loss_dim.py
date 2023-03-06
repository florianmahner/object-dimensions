import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from collections import defaultdict

# directories = ["./results/sslab_final/deep/vgg16_bn/classifier.3/20.mio/sslab/500/16384",
#                 "./results/sslab_behavior",
#                 "./results/sslab_final/deep/vgg16_bn/classifier.3/20.mio/sslab/500/256"]

directories = ["./results/sslab_behavior/behavior/4.58mio/sslab/300/256"]
names = ["VICE Behavior Batch Size 256"]

# names = ["VGG 16 Batch Size - 16384", "Behavior Batch Size - 256", "VGG 16 Batch Size - 256"]
# excludes = [0.0031, 0.0032, 0.0033]
excludes = []

for base_dir, name in zip(directories, names):
    files = glob.glob(
        os.path.join(base_dir, "**/params_epoch_1000.npz"), recursive=True
    )

    dims, val_losses = defaultdict(list), defaultdict(list)

    if len(files) == 0:
        print("No files found in {}".format(base_dir))
        continue

    for i, f in enumerate(files):
        params = np.load(f)
        beta = float(params["beta"])
        if beta in excludes:
            continue
        iden = str(beta)
        dims[iden].append(params["dim_over_time"][-1])
        val_losses[iden].append(params["val_loss"][-1])

    # Sort the dimensions and val_losses by the beta values
    dims = dict(sorted(dims.items(), key=lambda item: float(item[0])))
    val_losses = dict(sorted(val_losses.items(), key=lambda item: float(item[0])))

    # print the number of files for each beta
    for key in dims.keys():
        print(f"{key}: {len(dims[key])}")

    errors = {}
    for key in dims.keys():
        errors[key] = np.std(dims[key])
        dims[key] = round(np.median(dims[key]))
        val_losses[key] = np.median(val_losses[key])

    betas = list(dims.keys())
    dims = list(dims.values())
    val_losses = list(val_losses.values())
    errors = list(errors.values())

    # Make a dataframe with the dimensions, val_losses and betas
    df = pd.DataFrame({"dim": dims, "val_loss": val_losses, r"$\beta$": betas})

    # Make a seaborn 2d scatter plot with betas as labels
    fig, ax = plt.subplots()
    cmap = sns.color_palette("husl", len(dims))
    for i in range(len(dims)):
        # ax.errorbar(val_losses[i], dims[i], lw=1.2, yerr=errors[i], fmt="o", label=str(betas[i]), c=cmap[i])
        ax.errorbar(
            dims[i],
            val_losses[i],
            lw=1.2,
            xerr=errors[i],
            fmt="o",
            label=str(betas[i]),
            c=cmap[i],
        )
    # sns.scatterplot(x='dim', y='val_loss', hue=r'$\beta$', data=df, ax=ax)

    ax.legend(ncol=2, columnspacing=0.6)

    ax.set_title(name)
    ax.set_xlabel("# Dimensions")
    ax.set_ylabel("Validation loss")

    name = name.split(" ")
    name = name[0] + "-" + name[-1]
    path = os.path.join("./results/plots", name + ".png")

    # Make a vertical line at at the dim location with the minimum of the validation loss
    min_val = df["val_loss"].min()
    max_val = df["val_loss"].max()
    max_dim = df[df["val_loss"] == max_val]["dim"].values[0]
    min_dim = df[df["val_loss"] == min_val]["dim"].values[0]

    # FInd the beta value for this minimum
    min_beta = df[df["val_loss"] == min_val][r"$\beta$"].values[0]
    # plt.axvline(x=min_val, color='black', linestyle='--', linewidth=0.4)
    plt.axvline(x=min_dim, color="black", linestyle="--", linewidth=0.4)
    # Write a text to the right of the line saying this is the minimum
    plt.text(
        min_dim + 2,
        max_val - (max_val - min_val) / 1.5,
        f"Optimal Model \n n = {min_dim}",
        fontsize=10,
        rotation=45,
    )
    # plt.text(min_val, max_dim - (max_dim  - min_dim) / 2 ,  f"Optimal Model \n n = {min_dim}", fontsize=10, rotation=45)

    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()
