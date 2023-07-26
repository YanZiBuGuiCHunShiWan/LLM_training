from __future__ import annotations
import abc,jsonlines
import transformers
from typing_extensions import NotRequired ,TypedDict

__all__=[
    "TokenizedSample",
    "CustomDatasets"
]

class TokenizedSample(TypedDict,total=True):
    input_ids: list
    labels: list
    attention_mask: list

class CustomDatasets(object):
    
    def __init__(self,tokenizer: transformers.AutoTokenizer,Max_seq_len: int):
        self.tokenizer=tokenizer
        self.Max_seq_len=Max_seq_len
        
    def tokenize(self,prompt,add_eos_token=True)->TokenizedSample:
        result = self.tokenizer(
        prompt,
        truncation=True,
        max_length=self.Max_seq_length,
        padding=False
    )
        if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.Max_seq_length
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if add_eos_token and len(result["input_ids"]) >= self.Max_seq_length:
            result["input_ids"][self.Max_seq_length - 1] = self.tokenizer.eos_token_id
            result["attention_mask"][self.Max_seq_length - 1] = 1

        result["labels"] = result["input_ids"].copy()
        return result
    
    def gen_from_jsonlines(self,datapath):
        with jsonlines.open(datapath,"r") as f:
            for line in f:
                yield line
                
    @abc.abstractmethod            
    def filter_fn(self):
        '''
        Rules to filter the samples that you don't need 
        '''
        raise NotImplementedError
                
    @abc.abstractmethod
    def generate_train_test_data(self):
        
        raise NotImplementedError