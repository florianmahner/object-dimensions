# %%

from scipy.io import loadmat
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

models_to_dimensions = {
    1: "Human",
    2: "Resnet50",
    3: "CLIP",
    4: "Densenet",
    5: "Barlow-Twins",
    6: "VGG16",
}

answer_to_quality = {
    1: "visual",
    2: "semantic",
    3: "mixed visual-semantic",
    4: "unclear",
}

answer_to_concept = {
    1: "single concept",
    2: "multiple concepts",
    3: "uninterpretable",
}


# %%


ratings = loadmat("/LOCAL/fmahner/object-dimensions/data/misc/dimension-ratings.mat")[
    "ratings"
]

models = ratings[:, 0]
qualities = ratings[:, 1]
concepts = ratings[:, 3]

# Extracting columns from the ratings array
models = ratings[:, 0]
qualities = ratings[:, 1]
concepts = ratings[:, 3]

df = pd.DataFrame({"Model": models, "Quality": qualities, "Concept": concepts})


df["Model"] = df["Model"].map(models_to_dimensions)
df["Quality"] = df["Quality"].map(answer_to_quality)

# remove unclear
df = df[df["Quality"] != "unclear"]


# Reorder models to have Human and VGG16 on top
model_order = ["Human", "VGG16", "Resnet50", "CLIP", "Densenet", "Barlow-Twins"]
df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)

# Counting occurrences of each quality per model and convert to percentage
quality_counts = df.groupby(["Model", "Quality"]).size().unstack(fill_value=0)
quality_percent = quality_counts.div(quality_counts.sum(axis=1), axis=0).reset_index()

# Plotting the stacked bar plot using seaborn and matplotlib
fig, ax = plt.subplots(figsize=(12, 8))

# Initialize the left position for stacking
left = pd.Series([0] * len(quality_percent), index=quality_percent.index)

# List of qualities for consistent ordering
qualities_list = ["visual", "mixed visual-semantic", "semantic"]

# Default seaborn color palette
palette = sns.color_palette()

# Map specific colors to each quality using default palette
quality_to_color = {
    "visual": palette[0],  # dark blue
    "mixed visual-semantic": palette[1],  # light blue
    "semantic": palette[2],  # orange
    # "unclear": palette[2],  # grey
}

# Plot each quality as a separate bar
for quality in qualities_list:
    sns.barplot(
        y="Model",
        x=quality,
        data=quality_percent,
        ax=ax,
        label=quality,
        color=quality_to_color[quality],
        left=left,
    )
    left += quality_percent[quality]

# Add a legend and informative axis label
ax.legend(ncol=1, loc="best", frameon=True)
ax.set(xlabel="Percentage of Ratings", ylabel="Model")
sns.despine(left=True, bottom=True)

plt.tight_layout()
plt.show()
# %%

# Do the same for the concept ratings
df["Concept"] = df["Concept"].map(answer_to_concept)

# Counting occurrences of each concept per model and convert to percentage
concept_counts = df.groupby(["Model", "Concept"]).size().unstack(fill_value=0)
concept_percent = concept_counts.div(concept_counts.sum(axis=1), axis=0).reset_index()

# Plotting the stacked bar plot for concept ratings using seaborn and matplotlib
fig, ax = plt.subplots(figsize=(12, 8))

# Initialize the left position for stacking
left = pd.Series([0] * len(concept_percent), index=concept_percent.index)

# List of concepts for consistent ordering
concepts_list = ["single concept", "multiple concepts", "uninterpretable"]

# Default seaborn color palette for concepts
palette_concepts = sns.color_palette()

# Map specific colors to each concept using default palette
concept_to_color = {
    "single concept": palette_concepts[0],  # blue
    "multiple concepts": palette_concepts[1],  # green
    "uninterpretable": palette_concepts[2],  # red
}

# Plot each concept as a separate bar
for concept in concepts_list:
    sns.barplot(
        y="Model",
        x=concept,
        data=concept_percent,
        ax=ax,
        label=concept,
        color=concept_to_color[concept],
        left=left,
    )
    left += concept_percent[concept]

# Add a legend and informative axis label
ax.legend(ncol=1, loc="best", frameon=True)
ax.set(xlabel="Percentage of Ratings", ylabel="Model")
sns.despine(left=True, bottom=True)

plt.tight_layout()
plt.show()
# %%
