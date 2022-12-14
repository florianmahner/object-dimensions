#!/bin/bash

vis=false
sparse_codes=false
big_gan=false
searchlight=false
gpt3=false

while getopts "vcgbs" opt; do
  case $opt in
    v) vis=true ;;
    c) sparse_codes=true ;;
    b) big_gan=true ;;
    g) gpt3=true ;;
    s) searchlight=true ;;
    \?) echo "Invalid option -$OPTARG" >&2 ;;
  esac
done


if $vis; then
    echo "Visualize the image maximally activating embedding dimensions"
    python deep_embeddings/analyses/visualization/visualize_embedding.py \
        --config "./configs/interpretability.toml" --section "visualization"   
fi

if $sparse_codes; then
    echo "Learn the sparse code predictions using RidgeRegression CV"
    sleep 2
    python deep_embeddings/analyses/sparse_codes/sparse_codes_ridge.py \
        --config "./configs/interpretability.toml" --section "sparse_codes"
fi

if $gpt3; then
    echo "Label dimensions using GPT3"
    python deep_embeddings/analyses/gpt3/extract_feature_norms.py \
        --config "./configs/interpretability.toml" --section "gpt3"
fi

if $big_gan; then
    echo "Generate images from the embedding using BigGAN"
    python deep_embeddings/analyses/image_generation/optimize_and_sample.py 
        --config "./configs/interpretability.toml" --section "big_gan"
fi

if $searchlight
then
    echo "Searchlight analysis"
    python deep_embeddings/analyses/searchlight/searchlight_one_image.py 
        --config "./configs/interpretability.toml" --section "searchlight"
fi
