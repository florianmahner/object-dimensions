#!/bin/bash

# Train the model
echo "Start gamma analysis with fixed batch size"

# General parameters
log_path="./results"
modality="deep" 
load_model=False
fresh=False
tensorboard=True
identifier="gamma_analysis"

# Model parameters
init_dim=100
prior='sslab'
modality="deep"
stability_time=2000
rnd_seed=42
batch_size=16396
n_epochs=50
params_interval=200
checkpoint_interval=300  # we dont save the model in principle!


# Data paths
feature_path='./data/vgg_bn_features_12/features.npy'
triplet_path='./data/triplets/triplets_12_1mio_pos_seed_0/'


# for gamma in `seq 0.2 0.025 0.5`
for gamma in `seq 0.4 0.02 0.5`
do 
    log_path_gamma=$log_path"_gamma_"$gamma

    echo "Start training the model with gamma = $gamma in $log_path_gamma"

    
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
        --log_path $log_path \
        --identifier $identifier & # This puts the process in the background
done

