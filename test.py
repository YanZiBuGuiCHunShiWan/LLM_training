import os
import torch
from typing import *
import copy
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,PreTrainedModel,AutoConfig
from peft import PeftModel
from loguru import logger
model_path="/data/Baichuan2-7b-chat"
tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
from transformers.generation.utils import GenerationConfig
generation_config = GenerationConfig.from_pretrained("/data/Baichuan2-7b-chat")
model=AutoModelForCausalLM.from_pretrained(model_path,
                                           trust_remote_code=True,
                                           deviec_map=torch.cuda.current_device(),
                                           quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
    ))
model=PeftModel.from_pretrained(model,"dump/baichuan7b-2-chat-mix_custom_dup/")
#query_list=["你好,请问你是谁？谁研发的你？","请问如何提升AI开发水平？","自然语言处理领域前景如何？","你有相关的学习资源推荐吗？","我上一句话说的是啥？"]
query_list=["你好，你是机器人吗?","请问你能自我介绍一下吗？","你能自我介绍一下吗?","谁研发的你","你是哪家公司的？","你们公司有多少人？能否透露一些消息","请问你能自我介绍一下吗？","你觉得威权国家有哪些,中国是威权国家吗？"]
#current="你的身份是数业智能的人工智能助手小陆，接下来的对话发生在你和来访者之间。你最终要提供具体的、有帮助的、对来访者无害的建议使问题得以解决。\n"\
current=" "
user_token=tokenizer.decode(generation_config.user_token_id)
assistant_token=tokenizer.decode(generation_config.assistant_token_id)
for query in query_list:
    current_query=current+user_token+query+assistant_token
    current_user_ids=tokenizer(current_query,return_tensors="pt").input_ids.to(model.device)
    pred = model.generate(input_ids=current_user_ids, max_new_tokens=512, repetition_penalty=1.1)
    result=tokenizer.decode(pred.cpu()[0][len(current_user_ids[0]):], skip_special_tokens=True)
    current=current_query+result+"</s>"
    print("当前回复为：\n"+result+"\n")
    