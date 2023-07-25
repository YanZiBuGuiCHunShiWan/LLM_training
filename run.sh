#!/usr/bin/env bash

export PYTHONPATH=$PWD
SFT_CONFIG_JSON="config/lora_config.json"
mkdir -p dump

CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node 3 train_sft.py --train_args_file $SFT_CONFIG_JSON
