#!/bin/bash

# Train the model
echo "Start training the model"
cd ..

# General parameters
# log_path=""./learned_embeddings/weights_vgg_12_1024bs_20mio_pos_gamma04_spike_0125_slab_05_pi_06"
log_path="../learned_embeddings/test"
modality="deep" 
load_model=True
fresh=False

# Model parameters
init_dim=100
prior='sslab'
modality="deep"
stability_time=2000
rnd_seed=42
batch_size=1024
n_epochs=8000
gamma=0.4
params_interval=200
checkpoint_interval=300  # we dont save the model in principle!
tensorboard=True


# Data paths
feature_path='./data/vgg_bn_features_12/features.npy'
triplet_path='./data/triplets_12_20mio_pos/'

echo $load_model
echo $fresh

python main.py \
    --log_path $log_path \ 
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
    --feature_path $feature_path \
    --triplet_path $triplet_path
