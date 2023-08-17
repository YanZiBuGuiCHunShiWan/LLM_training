import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from transformers.generation.utils import GenerationConfig
import os,datetime,uvicorn
import argparse
import uuid,json
from peft import (
    PeftModel
)
import warnings
from fastapi import FastAPI, Request
from transformers.generation.utils import logger
logger.setLevel("ERROR")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="/data/Firefly-baichuan-7b", 
                    choices=["../ptm/baichuan", 
                             "/data/Llama2-Chinese-7b-Chat/"], type=str)
#parser.add_argument("--lora_path",type=str,default="dump/firefly-baichuan-7b-prompt/")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_path=args.model_name
logger.info("Instatianting Model and tokenizer")
model=AutoModelForCausalLM.from_pretrained(model_path,
                                           trust_remote_code=True,
                                           torch_dtype=torch.float16,
                                           device_map=torch.cuda.current_device(),
                                           quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"))
tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
logger.info("Loading complete")
app = FastAPI()
history_mp = {} # restore history for every uid

@app.post("/")
async def create_item(request: Request):
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    query = json_post_list.get('prompt') #Human: +query+\n</s>
    uid = json_post_list.get('uid', None)
    if uid == None or not(uid in history_mp):
        uid = str(uuid.uuid4())
        history_mp[uid] = []
    prompt=""
    for i, (old_query, response) in enumerate(history_mp[uid]):
        prompt+= '<s>Human: ' + old_query + '\n</s><s>Assistant: ' + response + "\n</s>"
    prompt+="<s>Human: "+query+"\n</s><s>Assistant: "
    max_length = json_post_list.get('max_length', 512)
    top_p = json_post_list.get('top_p', 0.85)
    temperature = json_post_list.get('temperature', 0.7)
    inputs = tokenizer(prompt, return_tensors="pt")
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    inputs = tokenizer(prompt, return_tensors="pt",add_special_tokens=False).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids, 
            attention_mask=inputs.attention_mask, 
            max_length=max_length, 
            do_sample=True, 
            top_k=50, 
            top_p=top_p, 
            temperature=temperature,
            repetition_penalty=1.1, 
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip("\n")
    history_mp[uid] = history_mp[uid] + [(query, response)]
    answer = {
        "response": response,
        "history": history_mp[uid],
        "status": 200,
        "time": time,
        "uid": uid,
        "content_length":len(prompt)+len(response)
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    return answer
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=19324, workers=1)
    