#!/bin/bash

# Train the model
echo "Start training the model"

# General parameters
log_path="./results"
# identifier=""
identifier="test"


modality="deep" 
load_model=False
fresh=False


triplet_path='./data/models/vgg16_bn/classifier.3/triplets_20mio/'
# triplet_path="./data/triplets_behavior"


# Model parameters
init_dim=100
prior="gauss"
n_epochs=3000
stability_time=500
seed=0
batch_size=16384
# batch_size=256
beta=1.0
params_interval=50
checkpoint_interval=500  # we dont save the model in principle!
tensorboard=True
scale=0.25
lr=0.001
device_id=1
non_zero_weights=5
mc_samples=5


python deep_embeddings/main.py \
    --data_path $data_path \
    --triplet_path $triplet_path \
    --modality $modality \
    --tensorboard \
    --fresh \
    --init_dim $init_dim \
    --prior $prior \
    --stability_time $stability_time \
    --seed $seed \
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
    --identifier $identifier
