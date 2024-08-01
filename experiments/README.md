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

















