#!/bin/bash

# Train the model
echo "Start training the model"

# Define all possible values
# betas=$(seq 0.8 0.2 3.0)

betas=$(seq 0.4 0.1 1.8)


gpu_id=3
n_gpus=3

# Iterate over the sequence of gamma
for beta in $betas; do

    # Grep the current amount of free memory on the gpu
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $gpu_id | grep -Eo [0-9]+)

    echo "Free memory on GPU $gpu_id: $free_mem"

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
    echo "Start training run with beta: $beta on GPU $gpu_id"

 
    python deep_embeddings/main.py --config "./configs/train_deep.toml" --beta $beta --device_id $gpu_id & 
    # python deep_embeddings/main.py --config "./configs/train_behavior.toml" --beta $beta --device_id $gpu_id & 

    # Wait for the process to occupy gpu mem
    sleep 10

    # Kill all processes starting with a pid starting with 1 and ending with any numbers
    
done