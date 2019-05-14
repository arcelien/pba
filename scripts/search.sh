#!/bin/bash
export PYTHONPATH="$(pwd)"

cifar10_search() {
    local_dir="$PWD/results/"
    data_path="$PWD/datasets/cifar-10-batches-py"

    python pba/search.py \
    --local_dir "$local_dir" \
    --model_name wrn_40_2 \
    --data_path "$data_path" --dataset cifar10 \
    --train_size 4000 --val_size 46000 \
    --checkpoint_freq 0 \
    --name "cifar10_search" --gpu 0.15 --cpu 2 \
    --num_samples 16 --perturbation_interval 3 --epochs 200 \
    --explore cifar10 --aug_policy cifar10 \
    --lr 0.1 --wd 0.0005
}

svhn_search() {
    local_dir="$PWD/results/"
    data_path="$PWD/datasets/"

    python pba/search.py \
    --local_dir "$local_dir" --data_path "$data_path" \
    --model_name wrn_40_2 --dataset svhn \
    --train_size 1000 --val_size 7325 \
    --checkpoint_freq 0 \
    --name "svhn_search" --gpu 0.19 --cpu 2 \
    --num_samples 16 --perturbation_interval 3 --epochs 160 \
    --explore cifar10 --aug_policy cifar10 --no_cutout \
    --lr 0.1 --wd 0.005
}

if [ "$1" = "rcifar10" ]; then
    cifar10_search
elif [ "$1" = "rsvhn" ]; then
    svhn_search
else
    echo "invalid args"
fi
