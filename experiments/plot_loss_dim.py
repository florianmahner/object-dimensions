import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

directories = ["./results/sslab_final/deep/vgg16_bn/classifier.3/20.mio/sslab/500/16384",
                "./results/sslab_behavior",  
                "./results/sslab_final/deep/vgg16_bn/classifier.3/20.mio/sslab/500/256"]

names = ["VGG 16 Batch Size - 16384", "Behavior Batch Size - 256", "VGG 16 Batch Size - 256"]

for base_dir, name in zip(directories, names):
    files = glob.glob(os.path.join(base_dir,"**/parameters.npz"), recursive=True)
    
    dims, val_losses, betas = [], [], []

    if len(files) == 0:
        print("No files found in {}".format(base_dir))
        continue

    for i, f in enumerate(files):
        params = np.load(f)
        dims.append(params['dim_over_time'][-1])
        val_losses.append(params['val_loss'][-1])
        betas.append(float(params['beta']))

    # Make a dataframe with the dimensions, val_losses and betas
    df = pd.DataFrame({'dim': dims, 'val_loss': val_losses, 'Regularization': betas})

    # Make a seaborn 2d scatter plot with betas as labels
    plt.figure()
    sns.scatterplot(x='dim', y='val_loss', hue='Regularization', data=df)
    plt.title(name)
    plt.xlabel("Dimension")
    plt.ylabel("Validation loss")
    name = name.split(" ")
    name = name[0] + "-" + name[-1]
    path = os.path.join("./experiments/plots", name + ".png")

    # Make a vertical line at at the dim location with the minimum of the validation loss 
    min_val = df['val_loss'].min()
    min_dim = df[df['val_loss'] == min_val]['dim'].values[0]
    plt.axvline(x=min_dim, color='black', linestyle='--', linewidth=1.0)
    
    # Write a text to the right of the line saying this is the minimum
    plt.text(min_dim + 5, min_val + 0.01, "Optimal Model", fontsize=10, rotation=45)

    plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
