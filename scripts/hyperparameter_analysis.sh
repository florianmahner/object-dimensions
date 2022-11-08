#!/bin/bash

echo "Start analysis with fixed batch size for a variety of hyperparams"

# Data paths
log_path="./results"
identifier="hyperparam_analysis"
data_path='./data/models'
triplet_path='./data/models/vgg16_bn/classifier.3/triplets_20mio'

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
# len of these combinations = 256
spikes=(0.125 0.25 0.5 1.0)
slabs=(0.25 0.5 1.0 2.0)
pis=(0.4 0.5 0.6)
gammas=(0.4 0.44 0.48 0.52 0.56 0.6)

# Build the cartesian product of all these combinations!
combinations=()
for i in "${spikes[@]}"; do
    for j in "${slabs[@]}"; do
        for k in "${pis[@]}"; do
            for l in "${gammas[@]}"; do
                combinations+=("$i $j $k $l")
            done
        done
    done
done


gpu_id=0

# Iterate over all combinations
for i in "${!combinations[@]}"; do

    # Grep the current amount of free memory on the gpu
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $gpu_id | grep -Eo [0-9]+)

    # If there is not enough memory, wait until there is
    while [ $free_mem -lt 1000 ]; do
        echo $free_mem
        echo "Not enough memory on GPU $gpu_id. Waiting..."
        sleep 30
        free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $gpu_id | grep -Eo [0-9]+)
    done

    echo "Running experiment $i"
    spike=$(echo ${combinations[$i]} | cut -d' ' -f1)
    slab=$(echo ${combinations[$i]} | cut -d' ' -f2)
    pi=$(echo ${combinations[$i]} | cut -d' ' -f3)
    gamma=$(echo ${combinations[$i]} | cut -d' ' -f4)

    echo "Spike: $spike, Slab: $slab, Pi: $pi, Gamma: $gamma"

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
        --identifier $identifier &

    # wait for the process to occupy gpu mem
    sleep 20

done