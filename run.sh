#!/usr/bin/env bash
export PYTHONPATH=$PWD
SFT_CONFIG_JSON="config/sft/lora_config.json"
SFT_MULTI_TURN_CONFIG_JSON="config/sft/lora_config_conversation.json"
RM_CONFIG_JSON="config/rm/lora_config.json"
mkdir -p dump


#CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node 3 train_sft.py --train_args_file $SFT_CONFIG_JSON   两种方式都可以
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch train_sft.py --train_args_file $SFT_MULTI_TURN_CONFIG_JSON