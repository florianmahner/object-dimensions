#!/bin/bash

# Train the model
echo "Start training the model"
# cd ..

# General parameters
# log_path=""./learned_embeddings/weights_vgg_12_1024bs_20mio_pos_gamma04_spike_0125_slab_05_pi_06"
log_path="./results/16396_bs_50mio_pos_gamma02_default_sslab"
modality="deep" 
load_model=True
fresh=False

# Model parameters
init_dim=100
prior='sslab'
modality="deep"
stability_time=2000
rnd_seed=42
batch_size=16396
n_epochs=6000
gamma=0.2
params_interval=200
checkpoint_interval=500  # we dont save the model in principle!
tensorboard=True


# Data paths
feature_path='./data/vgg_bn_features_12/features.npy'
triplet_path='./data/triplets/triplets_12_50mio_pos/'


python main.py \
    --feature_path $feature_path \
    --triplet_path $triplet_path \
    --modality $modality \
    --load_model $load_model \
    --fresh False \
    --tensorboard $tensorboard \
    --init_dim $init_dim \
    --prior $prior \
    --stability_time $stability_time \
    --rnd_seed $rnd_seed \
    --batch_size $batch_size \
    --n_epochs $n_epochs \
    --gamma $gamma \
    --params_interval $params_interval \
    --checkpoint_interval $checkpoint_interval \
    --log_path $log_path 
