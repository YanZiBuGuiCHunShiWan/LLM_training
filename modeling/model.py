import os
import torch
from typing import *
import copy
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,PreTrainedModel,AutoConfig
from loguru import logger
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from trl import AutoModelForCausalLMWithValueHead
from config.constants import LAYERNORM_NAMES
from config.args import FinetuneArguments

check_min_version('4.29.1')

################ find modules that can be trained by lora #######################  
def find_all_linear_modules(model,quantization="4bit"):
    assert quantization in ["4bit","8bit",False,None]
    if quantization=="4bit":
        cls=bnb.nn.Linear4bit
    elif quantization=="8bit":
        cls=bnb.nn.Linear8bitLt
    else:
        cls=nn.Linear
    lora_target_modules=set()
    for names,module in model.named_modules():
        if isinstance(module,cls):
            names=names.split(".")
            lora_target_modules.add(names[0] if len(names)==1 else names[-1])
            
    if 'lm_head' in lora_target_modules: 
        lora_target_modules.remove('lm_head')
    if "summary" in lora_target_modules:
        lora_target_modules.remove("summary")
    if '0'in lora_target_modules:
        lora_target_modules.remove('0')
    return list(lora_target_modules)
##################################################################################

def prepare_model_for_training(
    model: PreTrainedModel,
    finetuning_type: str,
    output_embedding_layer_name: Optional[str] = "lm_head",
    use_gradient_checkpointing: Optional[bool] = True,
    layer_norm_names: Optional[List[str]] = LAYERNORM_NAMES
) -> PreTrainedModel:

    for name, param in model.named_parameters():
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)

    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model.gradient_checkpointing_enable()
        model.config.use_cache = False # turn off when gradient checkpointing is enabled

    if finetuning_type != "full" and hasattr(model, output_embedding_layer_name):
        output_embedding_layer: torch.nn.Linear = getattr(model, output_embedding_layer_name)
        input_dtype = output_embedding_layer.weight.dtype
        class CastOutputToFloat(torch.nn.Sequential):

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return super().forward(x.to(input_dtype)).to(torch.float32)

        setattr(model, output_embedding_layer_name, CastOutputToFloat(output_embedding_layer))

    return model
        

def load_model(finetune_args: FinetuneArguments,local_rank=None,inference=False):
    
    config=AutoConfig.from_pretrained(finetune_args.model_name_or_path,trust_remote_code=True)
    device_map={"":torch.cuda.current_device()}
    local_rank=os.environ.get('LOCAL_RANK',local_rank)
    
    if "baichuan" in finetune_args.model_type.lower():
        """Baichuan/Bloom/Moss/LLama"""
        logger.info("Model Type:.........Baichuan.............")
        AutoLoader=AutoModelForCausalLM
    elif "chatglm" in finetune_args.model_type.lower():
        AutoLoader=AutoModel
    else:
        raise NotImplementedError
    if local_rank:
        device_map={"":int(local_rank)}
        logger.info("local rank {} map {}".format(local_rank,device_map))        
        
    if finetune_args.quantization=="4bit":
        require_version('bitsandbytes>=0.37.0',
                        'To fix: pip install bitsandbytes>=0.37.0')
        logger.info("load model with 4bit quantization")
        model = AutoLoader.from_pretrained(
            finetune_args.model_name_or_path,
            config=config,
            device_map=device_map,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        )
    elif finetune_args.quantization=="8bit":
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
            finetune_args.model_name_or_path,
            config=config,
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
            finetune_args.model_name_or_path,
            config=config,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    
     # Enable model parallelism.
    # 设置两个和并行操作相关的参数
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    if not inference:
        model=prepare_model_for_training(model,finetuning_type=finetune_args.finetuning_type)
    if finetune_args.finetuning_type=="full":
        model=model.float()
    padding_side="right"
    if finetune_args.training_stage=="rm":
        padding_side="left"
        model=AutoModelForCausalLMWithValueHead.from_pretrained(model,trust_remote_code=True)
    
    tokenizer=AutoTokenizer.from_pretrained(finetune_args.model_name_or_path,
                                            trust_remote_code=True,
                                            padding_side=padding_side)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id=tokenizer.unk_token_id # set pad id to unk id,the original unk id is 0 .
    assert tokenizer.pad_token_id!=tokenizer.eos_token_id,"pad_token_id should not be equal to eos_token_id"
    return model,tokenizer

if __name__=="__main__":
    model_name_or_path="../../ptm/baichuan"
    model,tokenizer=load_model(model_name_or_path,"4bit",1)
    logger.info("pad id of current tokenizer is {}.".format(tokenizer.pad_token_id))
    logger.info("unk id of current tokenizer is {}.".format(tokenizer.unk_token_id))