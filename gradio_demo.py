import torch
import gradio as gr
import os
import time
import argparse
import bitsandbytes as bnb
from peft import PeftModel
from config.constants import PROMPT_DICT
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/data/Baichuan-13B-Chat",help="model checkpoint folder")
    parser.add_argument("--lora_path", default=None, help="lora checkpoint folder")
    parser.add_argument("--quantize", default=None, help="quantization config optional [None, 4bit, 8bit]")
    parser.add_argument("--type",default="single-turn",choices=["single-turn","multi-turn"],help="only single-turn and multi-turn are supported")
    args = parser.parse_args()
    return args


def load_model_and_tokenizer(args):
    config_kwargs={
        "trust_remote_code":True
    }
    if args.quantize=="4bit":
        config_kwargs["torch_dtype"]=torch.float16
        config_kwargs["quantization_config"]=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
    elif args.quantize=="8bit":
        config_kwargs["torch_dtype"]=torch.float16
        config_kwargs["load_in_8bit"]=True
        config_kwargs["quantization_config"]=BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
    else:
        config_kwargs["torch_dtype"]=torch.float16
    config_kwargs["device_map"]=torch.cuda.current_device()
    print(config_kwargs)
    model=AutoModelForCausalLM.from_pretrained(args.model_path,**config_kwargs)
    if args.lora_path:
        model=PeftModel.from_pretrained(model,args.lora_path)
    tokenizer=AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id=tokenizer.unk_token_id # set pad id to unk id,the original unk id is 0 .
    assert tokenizer.pad_token_id!=tokenizer.eos_token_id,"pad_token_id should not be equal to eos_token_id"
    return model,tokenizer

def predict(input, chatbot, max_length, top_p, temperature, history):
    chatbot.append((input,""))
    curr_prompt=PROMPT_DICT["prompt_user"].format(input)+PROMPT_DICT["prompt_assistant"].replace("{}","")
    
    ####################多轮对话######################
    history_prompt="<s>"
    if args.type=="multi-turn":
        for idx,(old_query,response) in enumerate(history):
            history_prompt+=PROMPT_DICT["prompt_input"].format(old_query,response)+"</s>"
        curr_prompt=history_prompt+curr_prompt
    ##################################################
    else:
    ######################单论对话####################
        curr_prompt=history_prompt+curr_prompt
    #################################################
    inputs=tokenizer(curr_prompt,return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs=model.generate(
            **inputs,
            max_length=max_length,
            top_p=top_p,
            temperature=temperature
        )
        response=tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    history=history+[(input,response)]
    chatbot[-1]=(input,response)
    print(f"chatbot is {chatbot}")
    print(f"history is {history}")
    return chatbot, history

def reset_state():
    return [], []

def reset_user_input():
    return gr.update(value='')

if __name__=="__main__":
    args=parse_arg()
    model,tokenizer=load_model_and_tokenizer(args)
    with gr.Blocks() as demo:
        conversation_type="单轮" if args.type=="single-turn" else "多轮"
        gr.HTML("""<h1 align="center">欢迎使用 Shuyeit baichuan 人工智能助手！\n我是一个{}对话模型</h1>""".format(conversation_type))

        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                        container=False)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                emptyBtn = gr.Button("Clear History")
                max_length = gr.Slider(
                    0, 1024, value=512, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.8, step=0.01,
                                label="Top P", interactive=True)
                temperature = gr.Slider(
                    0, 1, value=0.7, step=0.01, label="Temperature", interactive=True)

        history = gr.State([])  # (message, bot_message)
        submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history],
                        show_progress=True)
        submitBtn.click(reset_user_input, [], [user_input])

        emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)
    
    demo.launch()