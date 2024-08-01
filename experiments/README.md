## Experimental Analyses

We provide the main experiments of the paper in the `./experiments` folder. The experiments are divided into the following categories:

- **Dimension Rating Experiment**: Flask experiment to get human observers rate the intepretability of each dimension in the DNN embedding and to give human annotated labels of these dimensions

-- **DNN Experiments**: Interpretability analysis of the DNN embeddings (ie activation maximization, causal testing, grad cam)

-- **Jackknife**: Jackknife resampling to relate the dimensions of the human and DNN embedding back to behavioral decisions in the triplet task

-- **Human Labeling**: Human labeling of the dimensions of the DNN embeddings, categorizing them into visual, mxied visual semantic and semantic

-- **RSA**: All Representational Similarity Analyses done for the paper

To run the experiments, we have provided config files in the `/configs` folder and scripts to execute different configs in the `/scripts` folder. To run these experiments, you can use the following command:

### Interpretability Analyses


For grad cam
```bash
bash ./scripts/interpretability_analyses.sh --g
```

For activation maximization

```bash
bash ./scripts/interpretability_analyses.sh --a
```

For causal testing

```bash
bash ./scripts/interpretability_analyses.sh --c
```

### Human DNN comparison

For the RSA analyses
```bash
bash ./scripts/human_dnn_comparison.sh -r
```

For the jackknife analyses
```bash
bash ./scripts/human_dnn_comparison.sh -j
```

For the human-DNN direct comparison
```bash
bash ./scripts/human_dnn_comparison.sh -d
```















