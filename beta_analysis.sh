#!/bin/bash

# Train the model
echo "Start training the model"

# General parameters
log_path="./results"
identifier="sslab"

modality="deep" 
load_model=False
fresh=False

data_path="./data/models"


triplet_path='./data/models/vgg16_bn/classifier.3/triplets_20mio/'
model_name="vgg16_bn"
module_name="classifier.3"


# Model parameters
init_dim=100
prior="sslab"
modality="deep"
n_epochs=3000
stability_time=500
rnd_seed=0
batch_size=16384
beta=1.0
params_interval=500
checkpoint_interval=500  # we dont save the model in principle!
tensorboard=True
scale=0.5
lr=0.001
device_id=1
non_zero_weights=5
mc_samples=5


# Define all possible values
betas =$(0.6 0.2 2.0)


gpu_id=1
n_gpus=3


# Iterate over the sequence of gamma
for beta in $betas; do

    # Grep the current amount of free memory on the gpu
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $gpu_id | grep -Eo [0-9]+)

    # If there is not enough memory, wait until there is
    while [ $free_mem -lt 1000 ]; do
        echo "Not enough memory on GPU $gpu_id. Waiting..."

        # for Iterate over n gpus
        for j in $(seq 0 $((n_gpus-1))); do
            # Get the current amount of free memory on the gpu
            free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $j | grep -Eo [0-9]+)

            # If there is enough memory, use this gpu
            if [ $free_mem -gt 1000 ]; then
                gpu_id=$j
                break
            fi
            # Else wait 60 seconds and try again
            sleep 60
        done
    done


    echo "Beta: $beta"

    python deep_embeddings/main.py \
    --data_path $data_path \
    --triplet_path $triplet_path \
    --modality $modality \
    --tensorboard \
    --fresh \
    --init_dim $init_dim \
    --prior $prior \
    --stability_time $stability_time \
    --rnd_seed $rnd_seed \
    --batch_size $batch_size \
    --n_epochs $n_epochs \
    --beta $beta \
    --scale $scale \
    --lr $lr \
    --params_interval $params_interval \
    --checkpoint_interval $checkpoint_interval \
    --log_path $log_path \
    --module_name $module_name \
    --model_name $model_name \
    --device_id $device_id \
    --mc_samples $mc_samples \
    --non_zero_weights $non_zero_weights \
    --identifier $identifier > training.out & 

    # wait for the process to occupy gpu mem
    sleep 10


done