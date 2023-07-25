import torch
import os
import argparse
import transformers
from transformers import Trainer, TrainingArguments, HfArgumentParser, set_seed
from peft import (
    LoraConfig,
    get_peft_model,
    PromptEncoderConfig
    
)
from dataclasses import field, fields, dataclass
import bitsandbytes as bnb
from modeling.model import load_model,find_all_linear_modules
from utils.generate_data import OpenSourceDataGen
from loguru import logger
from typing import Dict, List, Tuple, Union,Optional
os.environ["CUDA_VISIBLE_DEVICES"]='1,2,3'

VAL_SET_SIZE=200

@dataclass
class FinetuneArguments:
    '''
    微调参数设置
    '''
    #通用参数设置
    peft_type: str = field()
    model_name_or_path: str = field()
    model_type: str = field()
    train_file: str = field(default="data/sft_data/Safetyprompts/typical_safety_scenarios.json")
    test_size: float = field(default=0.2)
    quantization: str = field(default=None)
    max_seq_length: int = field(default=1024)
    #LoRA参数设置
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[int] = field(default=0.1)
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    
    
def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='config/lora_config.json', help="")
    args = parser.parse_args()
    train_args_file = args.train_args_file
    # 读取训练的参数配置
    parser = HfArgumentParser((FinetuneArguments, TrainingArguments))
    # 解析得到自定义参数，以及自带参数
    args, training_args = parser.parse_json_file(json_file=train_args_file)
    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    set_seed(training_args.seed)
    return args, training_args


def main():
    args,training_args=setup_everything()
    set_seed(42)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 1))
    logger.info(f"world size {world_size} local rank {local_rank}")
    
    #################### wrap model with peft #####################
    model,tokenizer=load_model(args.model_type,args.model_name_or_path,args.quantization,local_rank)
    if args.peft_type=="lora":
        lora_target_modules=find_all_linear_modules(model)
        config=LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=lora_target_modules
        )
    model=get_peft_model(model,config)
    if local_rank==1:
        model.print_trainable_parameters()
    #################### prepare data for training ################
    datagenerator=OpenSourceDataGen(tokenizer=tokenizer,Max_seq_length=args.max_seq_length)
    train_dataset,valid_dataset=datagenerator.generate_train_test_data(datapath=args.train_file,field="all",test_size=args.test_size)
    
    #################### start training ###########################    
    trainer=Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        )
    )
    trainer.train(resume_from_checkpoint=False)
    model.save_pretrained(training_args.output_dir)


if __name__=="__main__":
    main()