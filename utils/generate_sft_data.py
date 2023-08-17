''' Data class for sft training'''
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import torch
import transformers
from datasets import load_dataset,Dataset
from torch.utils.data import DataLoader as DL,Dataset as DS
from loguru import logger
from transformers import AutoTokenizer
from config.constants import SAFETY_PROMPT_FIELDS,PROMPT_DICT
from utils.base import CustomDatasets,TokenizedSample
from typing_extensions import TypedDict
from transformers import DataCollatorWithPadding

__all__=[
    "OpenSourceDataGen",
    "MultiturnConversationCollatorwithPadding"
]


class MutiturnConversationCollatorWithPadding(object):
    def __init__(self,tokenizer,Max_seq_length):
        self.tokenizer=tokenizer
        self.Max_seq_length=Max_seq_length
        self.pad_token_id=self.tokenizer.pad_token_id
    
    def __call__(self,batch: List[Dict[str, Any]]) -> TokenizedSample:
        lengths = [len(x['input_ids']) for x in batch]
        batch_max_len = min(max(lengths), self.Max_seq_length)
        input_ids_batch, attention_mask_batch, target_mask_batch = [], [], []
        for x in batch:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            target_mask = x['labels']
            padding_len = batch_max_len - len(input_ids)
            
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            
            input_ids = input_ids[:self.Max_seq_length]
            attention_mask = attention_mask[:self.Max_seq_length]
            target_mask = target_mask[:self.Max_seq_length]

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)

        
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': target_mask_batch
        }
        return inputs


class OpenSourceDataGen(CustomDatasets):
    '''
    数据集制作思路：
        用load_dataset直接读取成datasets.arrow_dataset.Dataset
        然后再用map函数按照指定的方式处理数据得到dataset(根据需求选择是否过滤数据),dataset种每一条数据如下:
        {"input_ids":xxxxxx,"attention_mask":xxxxxx,"labels":xxxxxx}
    '''
    @staticmethod    
    def is_contains_chinese(strs):
        for _char in strs:
            if '\u4e00' <= _char <= '\u9fa5':
                return True
        return False

    def generate_prompt_and_tokenize(self,data_dict):
        input=""
        if "response" in data_dict.keys():
            prompt=data_dict["prompt"]
            answer=data_dict["response"]
        elif "target" in data_dict.keys():
            prompt=data_dict["input"]
            answer=data_dict["target"]
        else:
            prompt=data_dict["instruction"]
            input=data_dict["input"]
            answer=data_dict["output"]
       
        full_prompt= self.tokenizer.bos_token+PROMPT_DICT["promt_input"].format((prompt+input),answer)+self.tokenizer.eos_token
        tokenized_full_prompt=self.tokenize(prompt=full_prompt)
        return tokenized_full_prompt
    
    def filter_fn(self,example):
        # 根据自己的需求编写过滤逻辑
        state=self.is_contains_chinese(example["conversation"][0]["human"])
        return state
    
    def generate_multiturn_tokenize(self,data_dict):
        utterances=[]
        for dict_info in data_dict["conversation"]:
            utterances.append(PROMPT_DICT["prompt_user"].format(dict_info["human"])+PROMPT_DICT["prompt_assistant"].replace("{}",""))
            utterances.append(dict_info["assistant"])
        utterances_ids=self.tokenizer(utterances,
                                      add_special_tokens=False,
                                      padding=False).input_ids
        input_ids=[self.tokenizer.bos_token_id]
        target_mask=[0]
        for i,utterances_ids in enumerate(utterances_ids):
            if i%2==0:
                target_mask+=[0]*(len(utterances_ids))
                input_ids+=utterances_ids
            else:
                target_mask+=[1]*(len(utterances_ids)+1)
                input_ids+=utterances_ids+[self.tokenizer.eos_token_id]
        assert len(input_ids)==len(target_mask)
        
        # 对长度进行截断
        input_ids = input_ids[:self.Max_seq_len]
        target_mask = target_mask[:self.Max_seq_len]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        labels=input_ids.copy() if not self.target_mask else target_mask
        return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        
    
    def __map_qiaoban(self,datadict):
        content=datadict["input"]
        input_dict=self.tokenizer(content,add_special_tokens=False,max_length=self.Max_seq_len,truncation=True)
        labels=input_dict["input_ids"].copy()
        input_dict["labels"]=labels
        return input_dict
        
        
    
    def __load_safety_prompts(self,data_path,field):
        assert field in SAFETY_PROMPT_FIELDS,"please check the field name."
        if field.lower()!="all":
            dataset=load_dataset("json",data_files=data_path,field=field,split="train")
        else:
            def tmp_gen():
                for field_name in SAFETY_PROMPT_FIELDS[:-1]:
                    datafile=load_dataset("json",data_files=data_path,field=field_name,split="train")
                    for j in datafile:
                        yield j #j: {"prompt":"xxx","response":"xxx","type":"xxx"}
            dataset=Dataset.from_generator(tmp_gen) #datasets.arrow_dataset.Dataset
        return dataset
        
    def __load_belle_cn_50(self,data_path):
        dataset=load_dataset("json",data_files=data_path)["train"]
        return dataset
    
    def load_qiaoban_conversation(self,datapath): 
        dataset=Dataset.from_generator(self.gen_from_jsonlines,gen_kwargs={"datapath":datapath})
        return dataset
    
    def __load_firefly(self,datapath):
        dataset=Dataset.from_generator(self.gen_from_jsonlines,gen_kwargs={"datapath":datapath})
        return dataset
    
    def __load_moss_sft(self,datapath):
        dataset=Dataset.from_generator(self.gen_from_jsonlines,gen_kwargs={"datapath":datapath}).filter(self.filter_fn)
        return dataset
        
    def generate_train_test_data(self,datapath,test_size,field="all"):
        
        map_func=self.generate_prompt_and_tokenize
        
        data_name=datapath.split("/")[-2]
        
        if data_name.lower()=="belle":
            logger.info("loading Belle.....")
            dataset=self.__load_belle_cn_50(datapath)
            column_names=dataset.column_names
            
        elif data_name.lower()=="firefly":
            logger.info("loading firefly........")
            dataset=self.__load_firefly(datapath)
            column_names=dataset.column_names
            
        elif data_name.lower()=="moss-sft":
            logger.info("loading moss-sft.........")
            dataset=self.__load_moss_sft(datapath)
            column_names=dataset.column_names
            map_func=self.generate_multiturn_tokenize
        
        elif data_name.lower()=="safetyprompts":
            logger.info("Loading safetyprompts......")
            dataset=self.__load_safety_prompts(datapath,field=field)
            column_names=["type","response","prompt"]
            
        elif data_name.lower()=="qiaoban":
            logger.info("Loading qiaoban........")
            dataset=self.load_qiaoban_conversation(datapath)
            map_func=self.__map_qiaoban
            column_names=dataset.column_names
        else:
            raise NotImplementedError
        
        if test_size>0:
            splitted_dataset=dataset.train_test_split(test_size=test_size,shuffle=False,seed=42) 
            train_dataset=splitted_dataset["train"].map(map_func,remove_columns=column_names,num_proc=8)
            valid_dataset=splitted_dataset["test"].map(map_func,remove_columns=column_names,num_proc=8)
        else:
            train_dataset=dataset.map(map_func,remove_columns=column_names,num_proc=8)
            valid_dataset=None
        return train_dataset,valid_dataset
    

if __name__=="__main__":
    #model_path="/data/Llama-2-7b-hf"
    model_path="../ptm/baichuan"
    tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    tokenizer.padding_side="right"
    tokenizer.pad_token_id=tokenizer.unk_token_id
    print("eos token id:",tokenizer.eos_token_id)
    print("bos token id: ",tokenizer.bos_token_id)
    #datafile="../data/Safetyprompts/typical_safety_scenarios.json"
    datafile="data/sft_data/Moss-sft/moss_tiny.jsonl"
    #datafile="../data/sft_data/Moss-sft/moss-003-sft-data.jsonl"
    #datafile="data/sft_data/Qiaoban/child_chat_data.json"
    datagenerator=OpenSourceDataGen(tokenizer,4096,Target_mask=True)
    train_dataset,cal_dataset=datagenerator.generate_train_test_data(datapath=datafile,test_size=0.2)
    # print(len(train_dataset))
    print(train_dataset[11])
    print(tokenizer.decode(train_dataset[11]["input_ids"]))
    print(train_dataset)
    Dataloader=DL(train_dataset,batch_size=3,collate_fn=transformers.DataCollatorForSeq2Seq(
           tokenizer=tokenizer,
           pad_to_multiple_of=8,
           return_tensors="pt",
       ))
    for data in Dataloader:
        print(data)
        print(data["input_ids"].shape)
        break
