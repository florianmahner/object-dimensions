# %%

from scipy.io import loadmat
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json


def plot_quality(df):
    df = df[df["Quality"] != "unclear"]

    # Counting occurrences of each quality per model and convert to percentage
    quality_counts = df.groupby(["Model", "Quality"]).size().unstack(fill_value=0)
    quality_percent = quality_counts.div(
        quality_counts.sum(axis=1), axis=0
    ).reset_index()

    # Plotting the stacked bar plot using seaborn and matplotlib
    fig, ax = plt.subplots(figsize=(6, 4))

    # Initialize the left position for stacking
    left = pd.Series([0] * len(quality_percent), index=quality_percent.index)

    # List of qualities for consistent ordering
    qualities_list = ["visual", "mix visual-semantic", "semantic"]

    # Default seaborn color palette
    palette = sns.color_palette("deep")

    # Map specific colors to each quality using default palette
    quality_to_color = {
        "visual": palette[0],  # dark blue
        "mix visual-semantic": palette[1],  # light blue
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

    # put the legend on top of the plot with, eg similar to a title and with pad inches
    ax.legend(
        ncol=len(qualities_list),
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 1.15),
    )

    ax.set(xlabel="Percentage of Ratings", ylabel="Model")
    sns.despine(left=True, bottom=True)
    fig.savefig(
        "/LOCAL/fmahner/object-dimensions/results/plots/dimension_quality_ratings.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def plot_concept(df):
    # Counting occurrences of each concept per model and convert to percentage
    concept_counts = df.groupby(["Model", "Concept"]).size().unstack(fill_value=0)
    concept_percent = concept_counts.div(
        concept_counts.sum(axis=1), axis=0
    ).reset_index()

    # Plotting the stacked bar plot for concept ratings using seaborn and matplotlib
    fig, ax = plt.subplots(figsize=(6, 4))

    # Initialize the left position for stacking
    left = pd.Series([0] * len(concept_percent), index=concept_percent.index)

    # List of concepts for consistent ordering
    concepts_list = ["single concept", "multiple concepts", "uninterpretable"]

    # Default seaborn color palette for concepts
    palette_concepts = sns.color_palette("deep")

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
    # put the legend on top of the plot with, eg similar to a title and with pad inches
    ax.legend(
        ncol=len(concepts_list),
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 1.15),
    )
    ax.set(xlabel="Percentage of Ratings", ylabel="Model")
    sns.despine(left=True, bottom=True)
    fig.savefig(
        "/LOCAL/fmahner/object-dimensions/results/plots/dimension_concept_ratings.pdf",
        dpi=300,
        bbox_inches="tight",
    )


# %%

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
    3: "mix visual-semantic",
    4: "unclear",
}

answer_to_concept = {
    1: "single concept",
    2: "multiple concepts",
    3: "uninterpretable",
}

dimension_mapping = json.load(
    open("/LOCAL/fmahner/object-dimensions/data/misc/dimension_mapping.json")
)

# Replace the keys in the dimension mapping with the model names

new_names = {
    "barlow": "Barlow-Twins",
    "resnet": "Resnet50",
    "densenet": "Densenet",
    "vgg": "VGG16",
    "clip": "CLIP",
    "human": "Human",
}

new_mapping = {}
for key, val in dimension_mapping.items():

    new_key = new_names[key]
    new_mapping[new_key] = val

dimension_mapping = new_mapping


# %%
ratings = loadmat("/LOCAL/fmahner/object-dimensions/data/misc/dimension-ratings.mat")[
    "ratings"
]

models = ratings[:, 0]
qualities = ratings[:, 1]
concepts = ratings[:, 4]


df = pd.DataFrame({"Model": models, "Quality": qualities, "Concept": concepts})

df["Model"] = df["Model"].map(models_to_dimensions)
df["Quality"] = df["Quality"].map(answer_to_quality)
df["Concept"] = df["Concept"].map(answer_to_concept)


# Function to reorder DataFrame based on ordering dictionary
def reorder_df(df, ordering):
    # Ensure the dictionary keys and values are integers
    ordering = {int(k): int(v) for k, v in ordering.items()}

    # Create a list of row indices based on the values in the dictionary
    new_order = sorted(ordering, key=ordering.get)

    # Check if the DataFrame length matches the ordering length
    if len(new_order) > len(df):
        new_order = new_order[: len(df)]

    # Reorder the DataFrame using these indices
    return df.iloc[new_order].reset_index(drop=True)


# Split the DataFrame by model
models = df["Model"].unique()
reordered_dfs = []

for model in models:
    model_df = df[df["Model"] == model]
    reordered_model_df = reorder_df(model_df, dimension_mapping[model])
    reordered_dfs.append(reordered_model_df)

# Concatenate the reordered DataFrames back into a single DataFrame
reordered_df = pd.concat(reordered_dfs).reset_index(drop=True)
model_order = ["Human", "VGG16", "Resnet50", "CLIP", "Densenet", "Barlow-Twins"]
reordered_df["Model"] = pd.Categorical(
    reordered_df["Model"], categories=model_order, ordered=True
)

df = reordered_df
df["Dimension"] = df.groupby("Model").cumcount()

# Save the df
df.to_csv(
    "/LOCAL/fmahner/object-dimensions/data/misc/dimension_ratings_processed.csv",
    index=False,
)


plot_quality(df)
plot_concept(df)

# %%


human = df[df["Model"] == "Human"]
vgg16 = df[df["Model"] == "VGG16"]


# drop the last 20 dimensions of human and vgg16 and make a pie chart comparison of the quality ratings
human = human.iloc[0:20]
vgg16 = vgg16.iloc[0:20]

# Aggregate the data for the pie charts
human_counts = human["Concept"].value_counts()
vgg16_counts = vgg16["Concept"].value_counts()

sns.set_palette("deep")
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].pie(human_counts, labels=human_counts.index, autopct="%1.1f%%", startangle=140)
ax[0].set_title("Human Quality Ratings")
ax[1].pie(vgg16_counts, labels=vgg16_counts.index, autopct="%1.1f%%", startangle=140)
ax[1].set_title("VGG16 Quality Ratings")


# %%
