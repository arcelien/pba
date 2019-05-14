#!/bin/bash
export PYTHONPATH="$(pwd)"

# args: [model name] [lr] [wd]
eval_cifar100() {
  hp_policy="$PWD/schedules/rcifar10_16_wrn.txt"
  local_dir="$PWD/results/"
  data_path="$PWD/datasets/cifar-100-python"

  size=50000
  dataset="cifar100"
  name="eval_cifar100_$1" # has 8 cutout size

  python pba/train.py \
    --local_dir "$local_dir" --data_path "$data_path" \
    --model_name "$1" --dataset "$dataset" \
    --train_size "$size" --val_size 0 \
    --checkpoint_freq 25 --gpu 1 --cpu 4 \
    --use_hp_policy --hp_policy "$hp_policy" \
    --hp_policy_epochs 200 \
    --aug_policy cifar10 --name "$name" \
    --lr "$2" --wd "$3"
}

if [ "$@" = "wrn_28_10" ]; then
  eval_cifar100 wrn_28_10 0.1 0.0005
elif [ "$@" = "ss_96" ]; then
  eval_cifar100 shake_shake_96 0.01 0.0025
elif [ "$@" = "pyramid_net" ]; then
  eval_cifar100 pyramid_net 0.025 0.0005
else
  echo "invalid args"
fi
