from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from peft import PeftModel
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from loguru import logger

def main():
    #model_name = '/data/Firefly-baichuan-7b'
    model_name="../ptm/baichuan"
    model_name="/data/Llama-2-7b-hf"
    #lora_path="output/firefly-baichuan-7b/checkpoint-1500"
    #lora_path="dump/baichuan-7b-clm_prompt"
    #lora_path="dump/baichuan-7b-shiftmask-prompt"
    lora_path="dump/llama-7b-shiftmask-prompt"
    # model_name = 'YeungNLP/firefly-baichuan-7b'
    # model_name = 'YeungNLP/firefly-ziya-13b'
    # model_name = 'YeungNLP/firefly-bloom-7b1'

    device = 'cuda'
    max_new_tokens = 512    # 每轮对话最多生成多少个token
    history_max_len = 4096  # 模型记忆的最大token长度
    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.1

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        device_map=torch.cuda.current_device()
    )
    model=PeftModel.from_pretrained(model,lora_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False if model.config.model_type == 'llama' else True
    )
    # 记录所有历史记录
    history_token_ids = tokenizer('<s>', return_tensors="pt").input_ids

    # 开始对话
    utterance_id = 0    # 记录当前是第几轮对话，为了契合chatglm的数据组织格式
    user_input = input('User：')
    while True:
        utterance_id += 1
        if model.config.model_type == 'chatglm':
            user_input = '[Round {}]\n\n问：{}\n\n答：'.format(utterance_id, user_input)
        else:
            user_input = 'Human: {}\n</s>Assistant: '.format(user_input)
        user_input_ids = tokenizer(user_input, return_tensors="pt", add_special_tokens=False).input_ids
        history_token_ids = torch.concat((history_token_ids, user_input_ids), dim=1)
        model_input_ids = history_token_ids[:, -history_max_len:].to(model.device)
        #model_input_ids = history_token_ids[:, :].to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=model_input_ids, max_new_tokens=max_new_tokens, do_sample=True, top_p=top_p,
                temperature=temperature, repetition_penalty=repetition_penalty, eos_token_id=tokenizer.eos_token_id
            )
        model_input_ids_len = model_input_ids.size(1)
        response_ids = outputs[:, model_input_ids_len:]
        history_token_ids = torch.concat((history_token_ids, response_ids.cpu()), dim=1)
        response = tokenizer.batch_decode(response_ids,skip_special_tokens=True)
        print("Firefly：" + response[0].strip().replace('</s>', ""))
        user_input = input('User：')

if __name__ == '__main__':
    main()