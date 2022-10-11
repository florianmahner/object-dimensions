#!/bin/bash
# This script is used to analyze the results of the experiments.
# TODO Implement GPT3 analyses and searchlight extraction!

vis_embedding=true
sparse_codes=false
big_gan=false
searchlight=false
gpt3=false

# General parameters
# embedding_path="./results/weights_vgg_12_2048bs_20mio_pos_gamma04/params/pruned_q_mu_epoch_3000.txt"
embedding_path="./results/weights_vgg_12_1024bs_20mio_pos_gamma04_spike_0125_slab_05_pi_06/params/pruned_q_mu_epoch_1200.txt"
dnn_path="./data/vgg_bn_features_12/features.npy"
rnd_seed=42
device="gpu"

# Visualization parameters
per_dim=False

# Sparse codes parameters
k_folds=4

# Big GAN parameters
model_name="vgg16_bn"
module_name="classifier.3"
n_samples=1000
window_size=50
batch_size=1
truncation=0.4
top_k=16
sample_latents=True
max_iter=200
dim=(1 2 3 4)
alpha=1.0
beta=1.0

# Searchlight parameters
# TODO Implement this

if $vis_embedding; then
    echo "Plot the topk predictions from the embedding"
    python deep_embeddings/analyses/visualization/visualize_embedding.py \
        --embedding_path=$embedding_path \
        --modality=deep \
        --n_images=12 \
        --per_dim=$per_dim
fi

if $sparse_codes; then
    echo "Learn the sparse code predictions using RidgeRegression CV"
    kfolds=4
    sleep 2
    
    python deep_embeddings/analyses/sparse_codes/sparse_codes_ridge.py \
        --dnn_path $dnn_path \
        --embedding_path $embedding_path \
        --k_folds $k_folds 
fi

if $gpt3; then
    echo "Label dimensions using GPT3"
    python deep_embeddings/analyses/gpt3/extract_feature_norms.py \
        --embedding_path $embedding_path \
        --dnn_path $dnn_path 
        --feature_norm_path "./data/feature_norms/feature_object_matrix.csv"
fi

if $big_gan; then
    echo "Generate images from the embedding using BigGAN"
    # sleep 2
    python deep_embeddings/analyses/image_generation/optimize_and_sample.py \
        --embedding_path $embedding_path \
        --model_name $model_name \
        --module_name $module_name \
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

# if $searchlight
# then
#     echo "Searchlight analysis"
# fi