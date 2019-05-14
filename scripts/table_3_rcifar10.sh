#!/bin/bash
export PYTHONPATH="$(pwd)"

# args: [model name] [lr] [wd]
eval_rcifar10() {
  hp_policy="$PWD/schedules/rcifar10_16_wrn.txt"
  local_dir="$PWD/results/"
  data_path="$PWD/datasets/cifar-10-batches-py"

  size=4000
  dataset="cifar10"
  name="eval_rcifar10_$1"

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
  eval_rcifar10 wrn_28_10 0.05 0.005
elif [ "$@" = "ss_96" ]; then
  eval_rcifar10 shake_shake_96 0.025 0.0025
else
  echo "invalid args"
fi
