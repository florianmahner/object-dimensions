#!/bin/bash

vis=false
sparse_codes=false
big_gan=false
searchlight=false
gpt3=false
causal=false
style=false

while getopts "vcbgsay" opt; do
  case $opt in
    v) vis=true ;;
    c) sparse_codes=true ;;
    b) big_gan=true ;;
    g) gpt3=true ;;
    s) searchlight=true ;;
    a) causal=true ;;
    y) style=true ;;


    \?) echo "Invalid option -$OPTARG" >&2 ;;
  esac
done


if $vis; then
    echo "Visualize the image maximally activating embedding dimensions"
    python experiments/visualization/visualize_embedding.py \
        --config "./configs/interpretability.toml" --section "visualization"   
fi

if $sparse_codes; then
    echo "Learn the sparse code predictions using Ridge / ElasticNet Regression CV"
    sleep 2
    python experiments/sparse_codes/sparse_codes.py \
        --config "./configs/interpretability.toml" --section "sparse_codes"
fi

if $gpt3; then
    echo "Label dimensions using GPT3"
    python experiments/gpt3/extract_feature_norms.py \
        --config "./configs/interpretability.toml" --section "gpt3"
fi

if $big_gan; then
    echo "Generate images from the embedding using BigGAN"
    python experiments/image_generation/optimize_and_sample.py \
        --config "./configs/interpretability.toml" --section "big_gan"
fi

if $searchlight
then
    echo "Grad CAM"
    python experiments/relevance_maps/grad_cam.py \
        --config "./configs/interpretability.toml" # does not need a section
fi

if $causal
then
    echo "Causal analysis"
    python experiments/causal/causal_comparison.py \
        --config "./configs/interpretability.toml" 
fi

if $style
then
    echo "Style gan analysis"
    python experiments/image_generation/optimize_and_sample_stylegan.py \
        --config "./configs/interpretability.toml" --section "act_max"
fi
