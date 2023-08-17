#!/bin/bash
export PYTHONPATH=$PWD
SFT_CONFIG_JSON="config/sft/lora_config.json"
SFT_MULTI_TURN_CONFIG_JSON="config/sft/lora_config_conversation.json"
RM_CONFIG_JSON="config/rm/lora_config.json"

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 train_sft.py --train_args_file $SFT_MULTI_TURN_CONFIG_JSON
# CUDA_VISIBLE_DEVICES=1,2 accelerate launch train_sft.py --train_args_file $SFT_MULTI_TURN_CONFIG_JSON