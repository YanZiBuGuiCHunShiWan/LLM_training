import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
"""
使用该脚本，将lora的权重合并到base model中
"""
def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/data/Baichuan-13B-Chat",help="model checkpoint folder")
    parser.add_argument("--lora_path", default=None, help="lora checkpoint folder")
    parser.add_argument("--save_path",default=None,type=str,help="The path at which you want to save the model")
    args = parser.parse_args()
    return args

def merge_lora_to_base_model(args):
    model_name_or_path = args.model_path
    adapter_name_or_path = args.lora_path
    save_path = args.save_path

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    model = PeftModel.from_pretrained(model, adapter_name_or_path)
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)


if __name__ == '__main__':
    args=parse_arg()
    merge_lora_to_base_model(args)
