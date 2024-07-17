#!/bin/bash

visualization=false
linear_model=false
grad_cam=false
causal=false
act_max=false

while getopts "vlgca" opt; do
  case $opt in
    v) visualization=true ;;
    l) linear_model=true ;;
    g) grad_cam=true ;;
    c) causal=true ;;
    a) act_max=true ;;
    \?) echo "Invalid option -$OPTARG" >&2 ;;
  esac
done


if $visualization; then
    echo "Run embedding visualization"
    python objdim/utils/visualization.py --config "./configs/interpretability_analyses.toml" --table "visualization"
fi

if $linear_model; then
    echo "Run linear model analyses"
    python experiments/dnn_experiments/linear_model.py --config "./configs/interpretability_analyses.toml" --table "linear_model"
fi

if $grad_cam; then
    echo "Run Grad-CAM analyses"
    python experiments/dnn_experiments/run_grad_cam.py --config "./configs/interpretability_analyses.toml" 
fi

if $causal; then
    echo "Run Causal analyses"
    python experiments/dnn_experiments/run_causal.py --config "./configs/interpretability_analyses.toml"
fi


if $act_max; then
    echo "Run Activation Maximization analyses"
    python experiments/dnn_experiments/run_activation_maximization.py --config "./configs/interpretability_analyses.toml" --table "act_max"
fi
