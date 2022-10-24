#!/bin/bash

echo "Start analysis with fixed batch size for a variety of hyperparams"

# Data paths
log_path="./results"
identifier="hyperparam_analysis"
data_path='./data/models'
triplet_path='./data/models/vgg16_bn/classifier.3/triplets_50mio'

# General parameters
modality="deep" 
load_model=False
fresh=False
tensorboard=False

# Model parameters
init_dim=100
prior='sslab'
modality="deep"
stability_time=1000
rnd_seed=42
batch_size=16384
n_epochs=3000
params_interval=1000
checkpoint_interval=3000  # we dont save the model in principle!n

# Define all possible values
spikes=(0,125 0.25 0.5 1.0)
slabs=(0.25 0.5 1.0 2.0)
pis=(0.3 0.4 0.5 0.6, 0.7)
gammas=(0.4 0.42 0.44 0.46 0.48 0.5 0.52 0.54 0.56 0.58 0.6)

# Iterate over all possible combinations (i.e. the cartesian grid!)
for spike in ${spikes[@]}; do

    for slabs in ${slabs[@]}; do

        for pi in ${pis[@]}; do

            for gamma in ${gammas[@]}; do

                echo "Start training with spike: $spike, slab: $slab, pi: $pi, gamma: $gamma"

                python deep_embeddings/main.py \
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
                    --spike $spike \
                    --slab $slab \
                    --pi $pi \
                    --log_path $log_path \
                    --identifier $identifier & # Submit jobs in parallel!

            done
        done
    done
