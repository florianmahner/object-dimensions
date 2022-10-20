#!/bin/bash

# Train the model
echo "Start training the model"
# cd ..

# TODO Implement this for bash or maybe do this in python scripts all to aggregate different models then?
model_comparison=false
seed_comparison=false

# General parameters
# log_path=""./learned_embeddings/weights_vgg_12_1024bs_20mio_pos_gamma04_spike_0125_slab_05_pi_06"
# log_path="./results/16396_bs_20mio_pos2_gamma04_default_sslab"
log_path="./results"
identifier="weibull"

modality="deep" 
load_model=False
fresh=False

# Data paths
data_path='./data/models/vgg16_bn'
triplet_path='./data/models/vgg16_bn/classifier.3/triplets_50mio/'

# Model parameters
init_dim=100
prior="weibull"
modality="deep"
stability_time=1000
rnd_seed=42
batch_size=16396
n_epochs=5000
gamma=0.25
params_interval=100
checkpoint_interval=500  # we dont save the model in principle!
tensorboard=True



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
