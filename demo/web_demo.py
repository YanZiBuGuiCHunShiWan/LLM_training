
import json
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from transformers.generation.utils import GenerationConfig
from loguru import logger
import bitsandbytes as bnb
import os
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel
    
)
MAX_LENGTH=10
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import loguru
st.set_page_config(page_title="Baichuan-chat")
st.title("Shuyeit@Baichuan-Chat-dev")
model_path="/data/Baichuan-13B-Chat"

@st.cache_resource
def init_model():
    loguru.logger.info("Instantiating..........................")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
        ),
        trust_remote_code=True
    )
    
    #model=PeftModel.from_pretrained(model,"dump/baichuan13b-chat-sft")
    loguru.logger.info("Start quantizing.........................")
    model.generation_config = GenerationConfig.from_pretrained(
        model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer


def clear_chat_history():
    del st.session_state.messages


def init_chat_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，我是百川大模型，很高兴为您服务🥰")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages
    

def main():
    model, tokenizer = init_model()
    messages = init_chat_history()

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
        if len(messages)<MAX_LENGTH:
            messages.append({"role": "user", "content": prompt})
        else:
            messages.pop(0)
            messages.append({"role": "user", "content": prompt})
        print(f"[user] {prompt}", flush=True)
        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            #inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            for response in model.chat(tokenizer, messages, stream=True):
                placeholder.markdown(response)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        messages.append({"role": "assistant", "content": response})
        print(json.dumps(messages, ensure_ascii=False), flush=True)

        st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()
