"""Dataset class for preference training."""

from __future__ import annotations
import torch
from typing import Any, Dict, Sequence
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from utils.base import CustomDatasets
from typing_extensions import TypedDict
from config.constants import PROMPT_DICT,SEED

__all__=[
    "PreferenceSample",
    "RMDataGen",
    "PairwiseDataCollatorWithPadding"
]


class PairwiseDataCollatorWithPadding(DataCollatorWithPadding):
    """
    Data collator for pairwise data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Pads batched data to the longest sequence in the batch.
        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        
        *** Make sure set the argument padding_side="left"  when instantiating the tokenizer***
        """
        features = [{"input_ids": feature[key]} for key in ("better_input_ids", "worse_input_ids") for feature in features]
        return super().__call__(features)

class PreferenceSample(TypedDict,total=True):
    
    accept_input_ids: list[int]
    reject_input_ids: list[int]


class RMDataGen(CustomDatasets):
    
    def generate_prompt_and_tokenize(self,data_dict)->PreferenceSample:
        start="<s>"
        intput_text=data_dict["instruction"]+data_dict["input"]
        accepted_answer=data_dict["output"][0]
        rejected_answer=data_dict["output"][1]
        accepted_text=start+PROMPT_DICT["prompt_input"].format(intput_text,accepted_answer)
        rejected_text=start+PROMPT_DICT["prompt_input"].format(intput_text,rejected_answer)
        accepted_dict=self.tokenize(prompt=accepted_text)
        rejected_dict=self.tokenize(prompt=rejected_text)
        
        return {
            "better_input_ids": accepted_dict["input_ids"],
            "worse_input_ids": rejected_dict["input_ids"],
        }
    
    def load_gpt_preference_data(self,datapath):
        dataset=load_dataset("json",data_files=datapath)["train"]
        return dataset
    
    
    def generate_train_test_data(self,datapath,test_size):
        dataset=self.load_gpt_preference_data(datapath)
        column_names=dataset.column_names
        if test_size>0:
            splitted_dataset=dataset.train_test_split(test_size=test_size,shuffle=False,seed=SEED)
            train_dataset=splitted_dataset["train"].map(self.generate_prompt_and_tokenize,remove_columns=column_names,num_proc=8)
            valid_dataset=splitted_dataset["test"].map(self.generate_prompt_and_tokenize,remove_columns=column_names,num_proc=8)
        else:
            train_dataset=dataset.map(self.generate_prompt_and_tokenize,remove_columns=column_names,num_proc=8)
            valid_dataset=None
        return train_dataset,valid_dataset
        
    
    
if __name__=="__main__":
    tokenizer=AutoTokenizer.from_pretrained("../../ptm/baichuan/",trust_remote_code=True,padding_side="left")
    tokenizer.pad_token_id=tokenizer.unk_token_id
    datafile="../data/RM_data/GPT4-llm/comparison_gpt4_data_zh.json"
    datagenerator=RMDataGen(tokenizer,1024)
    train_dataset,val_dataset=datagenerator.generate_train_test_data(datapath=datafile,test_size=0.2)
    print(tokenizer.bos_token)
    print(tokenizer.bos_token_id)
    print(tokenizer.eos_token)
    print(tokenizer.eos_token_id)
    Dataloader=DataLoader(train_dataset,batch_size=2,shuffle=False,collate_fn=PairwiseDataCollatorWithPadding(tokenizer=tokenizer,padding=True,return_tensors="pt"))
    for data in Dataloader:
        for line in data["input_ids"]:
            print(tokenizer.decode(line))
        #print(data["input_ids"][:,:8])
        #print(data["attention_mask"])
        break
    