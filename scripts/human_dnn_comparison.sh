#!/bin/bash

human_embedding_path="./results/weights_things_behavior_256bs/params/pruned_q_mu_epoch_3000.txt"
dnn_embedding_path="./results/16396_bs_50mio_pos_gamma02_default_sslab/params/pruned_q_mu_epoch_1000.txt"
feature_path="./data/vgg_bn_features_12"

python deep_embeddings/analyses/human_dnn/compare_human_dnn.py \
    --human_embedding_path=$human_embedding_path \
    --dnn_embedding_path=$dnn_embedding_path \
    --feature_path=$feature_path


