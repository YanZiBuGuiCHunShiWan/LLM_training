"""Dataset class for preference training."""

from __future__ import annotations
import jsonlines,transformers
import functools
from datasets import load_dataset,Dataset
from transformers import AutoTokenizer
from loguru import logger
from utils.base import CustomDatasets
from typing import Callable
from typing_extensions import TypedDict
from config.constants import PROMPT_DICT

__all__=[
    "PreferenceSample",
    "RMDataGen"
]

class PreferenceSample(TypedDict,total=True):
    
    accept_input_ids: list[int]
    accept_attention_mask: list[int]
    
    reject_input_ids: list[int]
    reject_attention_mask: list[int]


class RMDataGen(CustomDatasets):
    
    def generate_prompt_and_tokenize(self,data_dict)->PreferenceSample:
        insturction=data_dict["instruction"]
        input=data_dict["input"]
        accepted_output=data_dict["output"][0]
        rejected_output=data_dict["output"][1]
        
        raise NotImplementedError
    
    def load_gpt_preference_data(self,datapath):
        dataset=load_dataset("json",data_files=datapath)["train"]
        return dataset
    
    
    def generate_train_test_data(self,datapath,test_size):
        
        return None
    
    
if __name__=="__main__":
    pass