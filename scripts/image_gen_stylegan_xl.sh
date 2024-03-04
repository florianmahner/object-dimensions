#!/bin/bash

embedding_path="./results/50mio/16396/0.4/0.25/1.0/0.5/42/params/pruned_q_mu_epoch_5200.txt"
seed=42
device="cuda:0"

model_name="vgg16_bn"
module_name="classifier.3"
n_samples=2_000_000
window_size=50
batch_size=10
truncation=0.7
top_k=16
sample_dataset=True
find_topk=True
max_iter=200
dim=(1)
alpha=1.0
beta=1.0

python experiments/optimize_and_sample_stylegan.py \
        --embedding_path $embedding_path \
        --model_name $model_name \
        --module_name $module_name \
        --n_samples $n_samples \
        --window_size $window_size \
        --batch_size $batch_size \
        --truncation $truncation \
        --top_k $top_k \
        --seed $seed \
        --sample_dataset $sample_dataset \
        --find_topk $find_topk \
        --max_iter $max_iter \
        --dim ${dim[@]} \
        --alpha $alpha \
        --beta $beta \
        --device $device