#!/bin/bash
# This script is used to analyze the results of the experiments.

vis_embedding=true
sparse_codes=false
big_gan=false
searchlight=false

# General parameters
embedding_path="./results/weights_vgg_12_2048bs_20mio_pos_gamma04/params/pruned_q_mu_epoch_3000.txt"
dnn_path="./data/vgg_bn_features_12_features.npy"
rnd_seed=42
device="cuda:0"

# Get the grand parent directory of the embedding path
base_dir=$(dirname $(dirname $embedding_path))
# Create output directory to store the results in
results_path="$base_dir/analyses"
echo "Create new results dir " $results_path
mkdir -p $results_path

# Sparse codes parameters
kfolds=4

# Big GAN parameters
model_name="vgg16_bn"
module_name="classifier.3"
regression_path="./results/sparse_code_predictions"
n_samples=200000
window_size=50
batch_size=2
truncation=0.4
top_k=16
sample_latents=true
max_iter=200
dim=(1 2 3 4)
alpha=1.0
beta=1.0

# Searchlight parameters
# TODO Implement this

if $vis_embedding
then
    echo "Plot the topk predictions from the embedding"
    python deep_embeddings/analyses/visualization/visualize_embedding.py \
        --embedding_path=$embedding_path \
        --results_path=$results_path \
        --modality=deep \
        --n_images=12 \
        # --per_dim
fi


if $sparse_codes
then
    echo "Learn the sparse code predictions using RidgeRegression CV"
    kfolds=4

    python deep_embedding.analysis.sparse_codes.sparse_codes_ridge.py \
        --dnn_path $dnn_path \
        --embedding_path $embedding_path \
        --kfolds $kfolds \
        --results_path $results_path
fi


if $big_gan
then
    echo "Generate images from the embedding using BigGAN"
    # NOTE this is not yet nicely working with multiple dimensions
    python deep_embedding.analysis.big_gan.generate_images.py \
        --dnn_path $dnn_path \
        --embedding_path $embedding_path \
        --model_name $model_name \
        --module_name $module_name \
        --regression_path $regression_path \
        --n_samples $n_samples \
        --window_size $window_size \
        --batch_size $batch_size \
        --truncation $truncation \
        --top_k $top_k \
        --rnd_seed $rnd_seed \
        --sample_latents $sample_latents \
        --max_iter $max_iter \
        --dim ${dim[@]} \
        --alpha $alpha \
        --beta $beta \
        --device $device
fi

if $searchlight
then
    echo "Searchlight analysis"
fi