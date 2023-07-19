import torch
import os
import argparse
import transformers
from transformers import Trainer, TrainingArguments, HfArgumentParser, set_seed
from peft import (
    LoraConfig,
    get_peft_model,
    
)
from dataclasses import field, fields, dataclass
import bitsandbytes as bnb
from modeling.model import load_model
from utils.generate_data import OpenSourceDataGen
from loguru import logger
from typing import Dict, List, Tuple, Union
os.environ["CUDA_VISIBLE_DEVICES"]='1,2,3'

VAL_SET_SIZE=200

@dataclass
class FinetuneArguments:
    model_name_or_path: str = ""
    data_path: str = ""
    test_size: float = 0.2
    tuning_task: str = "safety_prompts"
    lora_rank: int = 8
    lora_target_modules: str = ""
    quantization: str = "4bit"
    model_output_dir: str = ""

def parse_args():
    parse=argparse.ArgumentParser(description="Arguments for model training.")
    parse.add_argument("--model_type",default="baichuan",type=str,choices=["chatglm","baichuan","moss"])
    parse.add_argument("--model_path",type=str,default="../ptm/baichuan")
    parse.add_argument("--model_output_dir",type=str,default="dump/baichuan6b-sft")
    parse.add_argument("--test_size",default=0.2,type=float)
    parse.add_argument("--lora_rank",default=8,type=int)
    parse.add_argument("--quantization",default="4bit",type=str)
    parse.add_argument("--data_path",default="data/Safetyprompts/typical_safety_scenarios.json",type=str)
    parse.add_argument("--lora_target_modules",default=None)
    args=parse.parse_args()
    return args

################ find modules that can be trained by lora ################  
def find_all_linear_modules(model):
    cls=bnb.nn.Linear4bit
    lora_target_modules=set()
    for names,module in model.named_modules():
        if isinstance(module,cls):
            names=names.split(".")
            lora_target_modules.add(names[0] if len(names)==1 else names[-1])
            
    if 'lm_head' in lora_target_modules: # needed for 16-bit
        lora_target_modules.remove('lm_head')
    return list(lora_target_modules)
        
def main():
    # args=HfArgumentParser(
    #     FinetuneArguments
    # ).parse_args_into_dataclasses
    args=parse_args()
    set_seed(42)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 1))
    logger.info(f"world size {world_size} local rank {local_rank}")
    
    #################### wrap model with peft #####################
    model,tokenizer=load_model(args.model_type,args.model_path,args.quantization,local_rank)
    lora_target_modules=args.lora_target_modules.splt(",") if args.lora_target_modules is not None else find_all_linear_modules(model)
    config=LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=lora_target_modules
    )
    model=get_peft_model(model,config)
    model.print_trainable_parameters()
    #################### prepare data for training ################
    
    datagenerator=OpenSourceDataGen(tokenizer=tokenizer)
    train_dataset,valid_dataset=datagenerator.generate_train_test_data(datapath=args.data_path,field="Mental_Health",test_size=args.test_size)
    
    #################### start training ###########################    
    trainer=Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=TrainingArguments(
            num_train_epochs=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=1,
            learning_rate=3e-4,
            gradient_accumulation_steps=4,
            evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
            save_strategy="steps",
            eval_steps=100 if VAL_SET_SIZE > 0 else None,
            save_steps=100,
            output_dir=args.model_output_dir,
            report_to = "tensorboard",
            save_total_limit=3,
            load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
            optim="adamw_torch",
            ddp_find_unused_parameters=False #如果启用多卡并行，得设置为False,如果gradient_checkpointing 是"used"
            ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        )
    )
    trainer.train(resume_from_checkpoint=False)
    model.save_pretrained(args.model_output_dir)


if __name__=="__main__":
    main()