#!/bin/bash
export PYTHONPATH=$PWD
export  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/anaconda3/lib/
SFT_CONFIG_JSON="config/sft/lora_config.json"
SFT_MULTI_TURN_CONFIG_JSON="config/sft/chatglm_empathy.json"
RM_CONFIG_JSON="config/rm/lora_config.json"

CUDA_VISIBLE_DEVICES=0,1,3,4  torchrun --nproc_per_node 4 train_sft.py --train_args_file $SFT_MULTI_TURN_CONFIG_JSON
# CUDA_VISIBLE_DEVICES=1,2 accelerate launch train_sft.py --train_args_file $SFT_MULTI_TURN_CONFIG_JSON
