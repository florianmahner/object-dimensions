#!/bin/bash
# This script is used to analyze the results of the experiments.
# TODO Implement GPT3 analyses and searchlight extraction!
    
vis_embedding=false
sparse_codes=false
big_gan=false
searchlight=true
gpt3=false

# General parameters
# embedding_path="./results/20mio/deep/0.4/0.25/1.0/0.5/42/params/pruned_q_mu_epoch_5400.txt"
embedding_path="./results/50mio/16396/0.4/0.25/1.0/0.5/42/params/pruned_q_mu_epoch_5200.txt"
# embedding_path="results/clip/clip/20mio/deep/sslab/100/16396/0.4/42/params/pruned_q_mu_epoch_400.txt"
# dnn_path="./data/models/clip/visual"

# embedding_path="./results/exp/vgg16_bn/20mio/deep/exp/100/16396/0.4/42/params/pruned_q_mu_epoch_100.txt"
dnn_path="./data/models/vgg16_bn/classifier.3"




# embedding_path="./results/test/20mio/deep/100/16396/0.4/0.25/1.0/0.5/42/params/pruned_q_mu_epoch_552.txt"
# embedding_path="./results/exp/vgg16_bn/50mio/deep/exp/100/16396/0.25/42/params/pruned_q_mu_epoch_900.txt"

# dnn_path="./data/models/clip/visual"
rnd_seed=42
device="cuda:0"

# Visualization parameters
per_dim=False
filter_images=False
# Sparse codes parameters
k_folds=4

feature_norm_path="./data/feature_norms/feature_object_matrix.csv"

# Big GAN parameters
model_name="vgg16_bn"
module_name="classifier.3"
n_samples=100000
window_size=50
batch_size=1
truncation=0.4
top_k=16
sample_latents=True
max_iter=200
dim=(0)
alpha=1.0
beta=1.0

# Searchlight parameters
image_root="./data/image_data/images12/"
feature_path="./data/models/vgg16_bn/classifier.3/"
window_size=20
stride=1
analysis="regression"

if $vis_embedding; then
    echo "Visualize the image maximally activating embedding dimensions"
    python deep_embeddings/analyses/visualization/visualize_embedding.py \
        --embedding_path=$embedding_path \
        --modality=deep \
        --n_images=12 \
        --per_dim=$per_dim \
        --filter_images=$filter_images
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
        --dnn_path $dnn_path \
        --feature_norm_path $feature_norm_path
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

if $searchlight
then
    echo "Searchlight analysis"
    python deep_embeddings/analyses/searchlight/searchlight_one_image.py \
        --embedding_path $embedding_path \
        --image_root $image_root \
        --analysis $analysis \
        --feature_path $feature_path \
        --model_name $model_name \
        --module_name $module_name \
        --window_size $window_size \
        --stride $stride 
fi