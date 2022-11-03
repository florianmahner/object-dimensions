#!/bin/bash

human_embedding_path="./results/behavior/vice/1.4mio/behavior/sslab/100/256/0.5/42/params/pruned_q_mu_epoch_3000.txt"
dnn_embedding_path="./results/50mio/16396/0.4/0.25/1.0/0.5/42/params/pruned_q_mu_epoch_5200.txt"
feature_path="./data/models_without_plus/vgg16_bn/classifier.3"

# python deep_embeddings/analyses/human_dnn/compare_human_dnn.py \
#     --human_embedding_path=$human_embedding_path \
#     --dnn_embedding_path=$dnn_embedding_path \
#     --feature_path=$feature_path

python deep_embeddings/analyses/human_dnn/compare_embeddings.py \
    --human_embedding_path=$human_embedding_path \
    --dnn_embedding_path=$dnn_embedding_path \
    --feature_path=$feature_path


