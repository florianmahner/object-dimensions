import numpy as np

import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt


palette = sns.color_palette("husl", 13)
alphabet = ["hallo", "hier", "bin", "ich", "und", "wer", "bist", "duuu", "langeslanges"]

df = pd.DataFrame({"weight": np.random.randint(0, 3, len(alphabet)), "Name": alphabet})

# # initialize the figure
plt.figure(figsize=(6, 8))
ax = plt.subplot(111, polar=True)
plt.axis("off")
# plt.title(o, fontsize = 27)

upperLimit = 7
lowerLimit = 0
width = 0.2

# Compute the angle each bar is centered on:
indexes = list(range(1, len(df.index) + 1))
width2 = 2 * np.pi / len(df.index)
angles = [element * width2 for element in indexes]


# Draw bars
bars = ax.bar(
    x=angles,
    height=df["weight"],
    width=0.2,
    bottom=lowerLimit,
    linewidth=2,
    edgecolor="white",
    color=palette,
)


# little space between the bar and the label
labelPadding = -0.5

# Add labels
for bar, angle, height, label in zip(bars, angles, df["weight"], df["Name"]):

    # Labels are rotated. Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle)

    # Flip some labels upside down
    alignment = ""
    if angle >= np.pi / 2 and angle < 3 * np.pi / 2:
        alignment = "right"
        rotation = rotation + 180
    else:
        alignment = "left"

    # Finally add the labels
    x_offset = angle - np.pi / 12
    y_offset = lowerLimit + bar.get_height() + labelPadding

    cut_off = 0.0
    if height > cut_off:
        ax.text(
            x=x_offset,
            y=y_offset,
            s=label,
            ha=alignment,
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=20,
        )

plt.savefig("rose.jpg")
