import os
import torch
from typing import *
import copy
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Seq2SeqTrainer,Seq2SeqTrainingArguments
from loguru import logger
from peft import prepare_model_for_kbit_training
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

check_min_version('4.29.1')

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
        

def load_model(model_type,model_name_or_path,quantization=None,local_rank=None):
    tokenizer=AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
    device_map={"":torch.cuda.current_device()}
    local_rank=os.environ.get('LOCAL_RANK',local_rank)
    if "baichuan" in model_type.lower():
        logger.info("Baichuan.............")
        AutoLoader=AutoModelForCausalLM
    elif "chatglm" in model_type.lower():
        AutoLoader=AutoModel
    else:
        raise NotImplementedError
    if local_rank:
        device_map={"":int(local_rank)}
        logger.info("local rank {} map {}".format(local_rank,device_map))        
        
    if quantization=="4bit":
        require_version('bitsandbytes>=0.37.0',
                        'To fix: pip install bitsandbytes>=0.37.0')
        logger.info("load model with 4bit quantization")
        model = AutoLoader.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        )
    elif quantization=="8bit":
        require_version('bitsandbytes>=0.39.0',
                        'To fix: pip install bitsandbytes>=0.39.0')
        require_version('transformers>=4.30.1',
                        'To fix: pip install transformers>=4.30.1')
        require_version('accelerate>=0.20.3',
                        'To fix: pip install accelerate>=0.20.3')
        require_version(
            'peft>=0.4.0.dev0',
            'To fix: pip install git+https://github.com/huggingface/peft.git')
        print("load model with 8bit quantization")
        model = AutoLoader.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        )
        
    else:
        logger.info("load model with bfloat16")
        model = AutoLoader.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    
     # Enable model parallelism.
    # 设置两个和并行操作相关的参数
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    if  quantization: #else full parameter training
        model=prepare_model_for_kbit_training(model)
        
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id=tokenizer.unk_token_id # set pad id to unk id,the original unk id is 0 .
    assert tokenizer.pad_token_id!=tokenizer.eos_token_id,"pad_token_id should not be equal to eos_token_id"
    return model,tokenizer

if __name__=="__main__":
    model_name_or_path="../../ptm/baichuan"
    model,tokenizer=load_model(model_name_or_path,"4bit",1)
    logger.info("pad id of current tokenizer is {}.".format(tokenizer.pad_token_id))
    logger.info("unk id of current tokenizer is {}.".format(tokenizer.unk_token_id))