#!/bin/bash

embedding_path="./results/50mio/16396/0.4/0.25/1.0/0.5/42/params/pruned_q_mu_epoch_5200.txt"
rnd_seed=42
device="cuda:0"

model_name="vgg16_bn"
module_name="classifier.3"
n_samples=1000000
window_size=50
batch_size=2
truncation=0.4
top_k=16
sample_latents=True
max_iter=200
dim=(0)
alpha=1.0
beta=1.0

python deep_embeddings/analyses/image_generation/optimize_and_sample_stylegan.py \
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