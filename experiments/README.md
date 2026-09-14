## Experimental Analyses

All experiments used in the paper are divided into the following categories:

- **Dimension Rating Experiment**: Flask experiment to get human observers rate the intepretability of each dimension in the DNN embedding and to give human annotated labels of these dimensions

- **DNN Experiments**: Interpretability analysis of the DNN embeddings (ie activation maximization, causal testing, grad cam)

- **Jackknife**: Jackknife resampling to relate the dimensions of the human and DNN embedding back to behavioral decisions in the triplet task

- **Human Labeling**: Human labeling of the dimensions of the DNN embeddings, categorizing them into visual, mxied visual semantic and semantic

- **RSA**: All Representational Similarity Analyses done for the paper

To run the experiments, we have provided config files in the [configs](../configs) folder and scripts to execute different configs in the [script](../script)  folder. Each of the `.toml` file in the configs folder corresponds to different
experiment settings and are the default settings used in the paper. Below is a list of commands to run the experiments in the paper.

### Interpretability Analyses

GradCam

```bash
bash ./scripts/interpretability_analyses.sh --g
```

Activation Maximization

If you want to reproduce the results of the activation maximization, you first need to download StyleGAN XL. We provide a script for this [here](../scripts/get_stylegan_xl.sh).

```bash
bash ./scripts/interpretability_analyses.sh --a
```

Causal Image Manipulations

```bash
bash ./scripts/interpretability_analyses.sh --c
```

### Human DNN comparison

RSA analyses
```bash
bash ./scripts/human_dnn_comparison.sh -r
```

Jackknife analyses
```bash
bash ./scripts/human_dnn_comparison.sh -j
```

Human-DNN direct comparison
```bash
bash ./scripts/human_dnn_comparison.sh -d
```

### Labeling

Human labeling
```python
python experiments/labeling/dnn_dimension_labeling.py --config configs/human_labeling.toml
```

Dimension ratings
```python
python experiments/labeling/dnn_dimension_ratings.py --config configs/human_labeling.toml
```


















### Expert-rating stimulus preparation and dimension mapping

The historical preparation script is now included as
[`scripts/behavioral_ratings_visualization.py`](../scripts/behavioral_ratings_visualization.py).
Its imports and embedding paths have been updated for the released repository,
and execution is guarded so importing it does not generate stimuli.
The original shuffling and visualization procedure is retained.

For each model, the script draws `np.random.permutation(W.shape[1])`,
reorders the embedding columns, and records a mapping from the displayed
zero-based dimension index to the original zero-based column index. JSON
serializes these index keys as strings. It also assigns anonymous model names
(`model_a` through `model_f`).

The outputs are saved under `results/plots/mixing_experiment_anonymous/`:

- `images/`: 9 × 9 grids of the 81 highest-weight images per dimension;
  entries with weights below 0.5 are replaced with gray images. These are
  top-weight grids, not percentile-sampled grids.
- `dimension_mapping.json`: the dimension permutations, keyed by model.
- `filenames_to_models.json`: original model names mapped to anonymous names.

The analysis reads the retained mapping at `data/misc/dimension_mapping.json`.
The preparation script writes to the results directory; it does not copy the
file into `data/misc`. Keep the released mapping with the released ratings.
The historical script does not set a random seed, so rerunning it generates
a new permutation and cannot reliably reconstruct the released mapping.

To prepare a new rating round, first download the embeddings and images and
check the input paths inside the script. Image order must match embedding row
order; the human image selection uses filenames containing `01b`. Run from
the repository root in the installed project environment:

```bash
poetry run python -m scripts.behavioral_ratings_visualization
```

The script retains its historical fixed paths and overwrites its output files
on reruns. Use a separate output location for each rating round and retain
the exact mapping with the corresponding stimuli and ratings. For another
dataset, adapt the input paths and image selection before running it.
