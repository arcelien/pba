#!/bin/bash
export PYTHONPATH="$(pwd)"

# args: [model name] [lr] [wd]
eval_svhn() {
  hp_policy="$PWD/schedules/rsvhn_16_wrn.txt"
  local_dir="$PWD/results/"
  data_path="$PWD/datasets/"
  name="eval_svhn_$1"
  size=1000
  dataset="svhn"
  python pba/train.py \
    --local_dir "$local_dir" --data_path "$data_path" \
    --model_name "$1" --dataset "$dataset" \
    --train_size "$size" --val_size 0 \
    --checkpoint_freq 0 --gpu 1 --cpu 4 \
    --use_hp_policy --hp_policy "$hp_policy" \
    --hp_policy_epochs 160 --epochs 160 \
    --aug_policy cifar10 --name "$name" \
    --lr "$2" --wd "$3" --no_cutout
}

if [ "$@" = "wrn_28_10" ]; then
  eval_svhn wrn_28_10 0.05 0.01
elif [ "$@" = "ss_96" ]; then
  eval_svhn shake_shake_96 0.025 0.005
else
  echo "invalid args"
fi
