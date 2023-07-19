import json,jsonlines
from datasets import load_dataset,Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader as DL,Dataset as DS
from transformers import default_data_collator

CUTOFF_LEN = 768
VAL_SET_SIZE = 2000
VAL_SIZE=0.2

class OpenSourceDataGen(object):
    def __init__(self,tokenizer: AutoTokenizer):
        self.tokenizer=tokenizer
        self.__Safetyprompts_fields=['Unfairness_And_Discrimination', 
                     'Crimes_And_Illegal_Activities', 
                     'Insult', 
                     'Mental_Health',
                     'Physical_Harm', 
                     'Privacy_And_Property',
                     'Ethics_And_Morality',
                     "all"]
        self.prompt={
            "moss":["<|MOSS|>","<eoh>\h<|HUMAN|>"],
            "baichuan":["Human: ","\n\nAssistant: "]
        }
        
    def tokenize(self,prompt,add_eos_token=True):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=CUTOFF_LEN,
            padding=False
        )
        if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < CUTOFF_LEN
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if add_eos_token and len(result["input_ids"]) >= CUTOFF_LEN:
            result["input_ids"][CUTOFF_LEN - 1] = self.tokenizer.eos_token_id
            result["attention_mask"][CUTOFF_LEN - 1] = 1

        result["labels"] = result["input_ids"].copy()
        return result
    
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
        input_text=self.tokenizer.bos_token+"Human: "+prompt+input+"\n\nAssitant: "
        full_prompt=input_text+answer
        tokenized_full_prompt=self.tokenize(prompt=full_prompt)
        return tokenized_full_prompt
    
    def __gen_from_jsonlines(self,datapath):
        with jsonlines.open(datapath,"r") as f:
            for line in f:
                yield line
    
    def load_safety_prompts(self,data_path,field):
        assert field in self.__Safetyprompts_fields,"please check the field name."
        if field.lower()!="all":
            dataset=load_dataset("json",data_files=data_path,field=field,split="train")
        else:
            def tmp_gen():
                for field_name in self.__Safetyprompts_fields[:-1]:
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
    
    def load_moss_sft(self,datapath):
        
        raise NotImplementedError
    
    def load_firefly(self,datapath):
        dataset=Dataset.from_generator(self.__gen_from_jsonlines,gen_kwargs={"datapath":datapath})
        return dataset
        
    def generate_train_test_data(self,datapath,test_size=0.2,field="all"):
        data_name=datapath.split("/")[-2]
        if data_name.lower()=="belle":
            dataset=self.load_belle_cn_50(datapath)
            column_names=None
        elif data_name.lower()=="firefly":
            dataset=self.load_firefly(datapath)
            column_names=dataset.column_names
        elif data_name.lower()=="moss-sft":
            raise NotImplementedError
        elif data_name.lower()=="safetyprompts":
            dataset=self.load_safety_prompts(datapath,field=field)
            column_names=["type","response","prompt"]
        else:
            raise NotImplementedError
        if test_size>0:
            splitted_dataset=dataset.train_test_split(test_size=test_size,shuffle=True,seed=42)
            train_dataset=splitted_dataset["train"].map(self.generate_prompt_and_tokenize,remove_columns=column_names,num_proc=4)
            valid_dataset=splitted_dataset["test"].map(self.generate_prompt_and_tokenize,remove_columns=column_names,num_proc=4)
        else:
            train_dataset=dataset.map(self.generate_prompt_and_tokenize,remove_columns=column_names,num_proc=4).shuffle()
            valid_dataset=None
        return train_dataset,valid_dataset
    
    
class CustomDataGen(DS,OpenSourceDataGen):
    def __init__(self) -> None:
        super().__init__()
        return None
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __len__(self):
        pass

if __name__=="__main__":
    tokenizer=AutoTokenizer.from_pretrained("../../ptm/baichuan/",trust_remote_code=True)
    #datafile="../data/Safetyprompts/typical_safety_scenarios.json"
    datafile="../data/Belle/Belle_open_source_0.5M.json"
    datagenerator=OpenSourceDataGen(tokenizer)
    train_dataset,cal_dataset=datagenerator.generate_train_test_data(datapath=datafile,test_size=0.2)
    print(tokenizer.bos_token)
    print(tokenizer.bos_token_id)
    print(tokenizer.eos_token)
    print(tokenizer.eos_token_id)
    
    print(train_dataset[0]["input_ids"])
    print(train_dataset[3]["input_ids"])
    print(train_dataset[9]["input_ids"])
    print(train_dataset[11]["input_ids"])