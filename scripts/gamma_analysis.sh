#!/bin/bash

# Train the model
echo "Start gamma analysis with fixed batch size"

# Data paths
log_path="./results"
identifier="gamma_analysis"
data_path='./data/models/vgg16_bn'
triplet_path='./data/models/vgg16_bn/classifier.3/triplets_1mio'

# General parameters
modality="deep" 
load_model=False
fresh=False
tensorboard=False

# Model parameters
init_dim=100
prior='sslab'
modality="deep"
stability_time=2000
rnd_seed=42
batch_size=2048
n_epochs=3
params_interval=5
checkpoint_interval=300  # we dont save the model in principle!n


# for gamma in `seq 0.2 0.025 0.5`
for gamma in `seq 0.4 0.04 0.5`
# for gamma in 0.4
do 
    log_path_gamma=$log_path"_gamma_"$gamma

    echo "Start training the model with gamma = $gamma in $log_path_gamma"

    python main.py \
        --data_path $data_path \
        --triplet_path $triplet_path \
        --modality $modality \
        --fresh \
        --tensorboard \
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
        --identifier $identifier 
done

