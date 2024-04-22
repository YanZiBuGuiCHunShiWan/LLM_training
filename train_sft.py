import os
import argparse
from transformers import TrainingArguments, HfArgumentParser, set_seed
from peft import (
    LoraConfig,
    get_peft_model
)
from modeling.model import load_model,find_all_linear_modules
from utils.generate_sft_data import OpenSourceDataGen,MultiturnConversationCollatorWithPadding
from modeling.model_trainer.sfttrainer import SFTTrainer
from loguru import logger
from config.args import FinetuneArguments

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
    custom_args,training_args=setup_everything()
    set_seed(42)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    logger.info(f"world size {world_size} local rank {local_rank}")
    
    #################### wrap model with peft #####################
    model,tokenizer=load_model(custom_args,local_rank)
        
    if custom_args.finetuning_type=="lora":
        lora_target_modules=find_all_linear_modules(model,quantization=custom_args.quantization)
        config=LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=custom_args.lora_rank,
            lora_alpha=custom_args.lora_alpha,
            lora_dropout=custom_args.lora_dropout,
            target_modules=lora_target_modules
        )
        model=get_peft_model(model,config)
    if local_rank==0:
        model.print_trainable_parameters()
    #################### prepare data for training ################
    datagenerator=OpenSourceDataGen(tokenizer=tokenizer,Max_seq_len=custom_args.max_seq_length)
    train_dataset,valid_dataset=datagenerator.generate_train_test_data(datapath=custom_args.train_file,
                                                                       test_size=custom_args.test_size,
                                                                       mask = 'collapsed',
                                                                       feedback_token=False)
    #################### start training ###########################    
    trainer=SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=training_args,
        data_collator=MultiturnConversationCollatorWithPadding(tokenizer,
                                                               custom_args.max_seq_length))
    trainer.train(resume_from_checkpoint=False)
    model.save_pretrained(training_args.output_dir)


if __name__=="__main__":
    main()