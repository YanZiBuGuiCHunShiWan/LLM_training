''' Data class for sft training'''
from __future__ import annotations
import jsonlines,transformers
from datasets import load_dataset,Dataset
from torch.utils.data import DataLoader as DL,Dataset as DS
from loguru import logger
from transformers import AutoTokenizer
from config.constants import SAFETY_PROMPT_FIELDS,PROMPT_DICT
from utils.base import CustomDatasets
from typing_extensions import TypedDict

__all__=[
    "OpenSourceDataGen"
]


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
        input_text=self.tokenizer.bos_token+"Human: "+prompt+input+"\n\nAssistant: "
        full_prompt=input_text+answer+self.tokenizer.eos_token
        tokenized_full_prompt=self.tokenize(prompt=full_prompt)
        return tokenized_full_prompt
    
    def filter_fn(self,example):
        # 根据自己的需求编写过滤逻辑
        Is_chinese=self.is_contains_chinese(example["conversation"][0]["human"])
        return Is_chinese  # 忽略 的样本
    
    def generate_multiturn_tokenize(self,data_dict):
        utterances=[]
        target_mask=[0]
        for dict_info in data_dict["conversation"]:
            utterances.append(PROMPT_DICT["prompt_user"].format(dict_info["human"])+PROMPT_DICT["prompt_assistant"].replace("{}",""))
            utterances.append(dict_info["assistant"]+"</s>")
        utterances_ids=self.tokenizer(utterances).input_ids
        input_ids=[self.tokenizer.bos_token_id]
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
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': target_mask
        }
        return inputs
    
    
    def load_safety_prompts(self,data_path,field):
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
    
        
    def load_belle_cn_50(self,data_path):
        dataset=load_dataset("json",data_files=data_path)["train"]
        return dataset
    
    def load_openassistant_conversation(self): 
        
        raise NotImplementedError
    
    def load_firefly(self,datapath):
        dataset=Dataset.from_generator(self.gen_from_jsonlines,gen_kwargs={"datapath":datapath})
        return dataset
    
    def load_moss_sft(self,datapath):
        dataset=Dataset.from_generator(self.gen_from_jsonlines,gen_kwargs={"datapath":datapath}).filter(self.filter_fn)
        return dataset
        
    def generate_train_test_data(self,datapath,test_size,field="all"):
        
        map_func=self.generate_prompt_and_tokenize
        
        data_name=datapath.split("/")[-2]
        
        if data_name.lower()=="belle":
            logger.info("loading Belle.....")
            dataset=self.load_belle_cn_50(datapath)
            column_names=dataset.column_names
        elif data_name.lower()=="firefly":
            logger.info("loading firefly........")
            dataset=self.load_firefly(datapath)
            column_names=dataset.column_names
            
        elif data_name.lower()=="moss-sft":
            logger.info("loading moss-sft.........")
            dataset=self.load_moss_sft(datapath)
            column_names=dataset.column_names
            map_func=self.generate_multiturn_tokenize
        
        elif data_name.lower()=="safetyprompts":
            logger.info("Loading safetyprompts......")
            dataset=self.load_safety_prompts(datapath,field=field)
            column_names=["type","response","prompt"]
            
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
    tokenizer=AutoTokenizer.from_pretrained("../../ptm/baichuan/",trust_remote_code=True)
    tokenizer.pad_token_id=tokenizer.unk_token_id
    #datafile="../data/Safetyprompts/typical_safety_scenarios.json"
    datafile="../data/sft_data/Moss-sft/moss-tiny.jsonl"
    datagenerator=OpenSourceDataGen(tokenizer,1024)
    train_dataset,cal_dataset=datagenerator.generate_train_test_data(datapath=datafile,test_size=0.2)
    print(tokenizer.bos_token)
    print(tokenizer.bos_token_id)
    print(tokenizer.eos_token)
    print(tokenizer.eos_token_id)
    
    print(train_dataset[0]["input_ids"])
    print(train_dataset[3]["input_ids"])
    print(train_dataset[9]["input_ids"])
    print(train_dataset[11]["input_ids"])
    print(tokenizer.decode(train_dataset[11]["input_ids"]))
    Dataloader=DL(train_dataset,collate_fn=transformers.DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        ))
    for data in Dataloader:
        print(data)
        break