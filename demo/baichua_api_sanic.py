import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from transformers.generation.utils import GenerationConfig
import os,datetime,uvicorn,asyncio
import argparse
import uuid,json
import warnings
from sanic import Sanic
from sanic.request import Request
from sanic.response import json
from loguru import logger

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="/data/Baichuan-13B-Chat", 
                    choices=["../ptm/baichuan", 
                             "/data/Llama2-Chinese-7b-Chat/"], type=str)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
app = Sanic(__name__)
history_mp = {} # restore history for every uid


@app.post("/test")
async def create_item(request: Request):
    print("hi")
    json_post_raw = request.json
    query = json_post_raw.get('prompt') 
    uid = json_post_raw.get('uid', None)
    if uid == None or not(uid in history_mp):
        uid = str(uuid.uuid4())
        history_mp[uid] = []
    if len(history_mp[uid])>=10:
        history_mp[uid]=history_mp[uid][-10:]
    messages=history_mp[uid]+[{"role": "user", "content": query}]
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    #response=await app.loop.run_in_executor(None,model.chat,tokenizer,messages)
    response =  model.chat(tokenizer,messages)
    history_mp[uid] = history_mp[uid] + [{"role": "assistant", "content": response}]
    answer = {
        "response": response,
        "history": history_mp[uid],
        "status": 200,
        "time": time,
        "uid": uid,
    }
    log = "[" + time + "] " + '", prompt:"' + query + '", response:"' + repr(response) + '"'
    print(log)
    return json(answer)

if __name__ == "__main__":
    uvicorn.run(app="main:myapp", host='0.0.0.0', port=19324, workers=1)
    