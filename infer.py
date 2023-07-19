import os
import torch
import argparse
from modeling.model import load_model
from transformers import  GenerationConfig
from peft import PeftModel
os.environ["CUDA_VISIBLE_DEVICES"]="1"
def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="../ptm/baichuan",help="model checkpoint folder")
    parser.add_argument("--lora_path", default=None, help="lora checkpoint folder")
    parser.add_argument("--quantize", default='4bit', help="quantization config optional [None, 4bit, 8bit]")
    args = parser.parse_args()
    return args

def main():
    args=parse_arg()
    model,tokenizer=load_model("baichuan",args.model_path,args.quantize,torch.cuda.current_device())
    if args.lora_path:
        model=PeftModel.from_pretrained(model,args.lora_path)
    model.eval()
    #input="“择偶”真的太令人痛苦了，可以根据财富来选择，而不是看脸色。"
    input="“在公司里，男性比女性更有领导力和决策力。"
    prompt="Human: "+input+"\n\nAsistant: "
    #prompt=input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_config = GenerationConfig(
        temperature=0.6,
        top_p=0.8,
        do_sample=True,
        repetition_penalty=2.0,
        max_new_tokens=512,  # max_length=max_new_tokens+input_sequence
    )
    generate_ids = model.generate(**inputs, generation_config=generation_config)
    output = tokenizer.decode(generate_ids[0][len(inputs.input_ids[0]):])
    print(output)

if __name__=="__main__":
    main()