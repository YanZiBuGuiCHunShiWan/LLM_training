''' Data class for sft training'''
from __future__ import annotations
from typing import Any, Dict, List,Union
import torch
from datasets import load_dataset,Dataset
from torch.utils.data import DataLoader as DL,Dataset as DS
from loguru import logger
from transformers import AutoTokenizer
from utils.base import CustomDatasets,TokenizedSample
from config.constants import FEEDBACK_TOKEN_DICT

__all__=[
    "OpenSourceDataGen",
    "MultiturnConversationCollatorwithPadding"
]

class MultiturnConversationCollatorWithPadding:
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
        labels = torch.where(target_mask_batch == 1, input_ids_batch, -100)
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': labels
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
    
    def filter_fn(self,example):
        # 根据自己的需求编写过滤逻辑
        state=self.is_contains_chinese(example["conversation"][0]["human"])
        return state
    
    def __build_chatglm(self,datadict,
                        mask: Union[str,"nomask","collapsed","ordinary"],
                        feedback_token:bool=False):
        '''
        mask: 损失函数计算掩码类型
            nomask->无掩码。
            collapsed->多轮对话拼接成一条，掩码掉用户提问部分。
            ordinary->多轮对话拆解成多条，掩码掉前k-1轮和当前第k轮的用户提问部分。
        feedback_token: 是否按照chain of hindsight 一样给回答添加feedback token，不计算feedback token的损失。
        '''
        gmask_token_id = self.tokenizer.get_command('[gMASK]')
        sop_token_id = self.tokenizer.get_command('sop')
        input_ids = [gmask_token_id, sop_token_id]  # 收集
        target_mask = [0] * 2
        if feedback_token:
            bad_feedback_token_ids=self.tokenizer(FEEDBACK_TOKEN_DICT["bad"],add_special_tokens=False).input_ids
            good_feedback_token_ids=self.tokenizer(FEEDBACK_TOKEN_DICT["good"],add_special_tokens=False).input_ids
            bad_feedback_token_len=len(bad_feedback_token_ids)
            good_feedback_token_len=len(good_feedback_token_ids)
            
        
        for dict_info in datadict["content"]:
            if feedback_token:
                user_token_ids = self.tokenizer.build_single_message("user","",dict_info["question"])
                bad_assistant_token_ids = self.tokenizer.build_single_message("assistant",
                                                                              "",
                                                                              FEEDBACK_TOKEN_DICT["bad"])
                bad_assistant_content_token_ids = self.tokenizer(dict_info["bad_answer"],add_special_tokens=False).input_ids+[self.tokenizer.eos_token_id]
                good_assistant_token_ids =  self.tokenizer(dict_info["good_answer"],add_special_tokens=False).input_ids+[self.tokenizer.eos_token_id]
                
                input_ids +=user_token_ids+bad_assistant_token_ids+bad_assistant_content_token_ids+good_feedback_token_ids+good_assistant_token_ids
                target_mask+=[0]*len(user_token_ids)+[0,1,1]+[0]*bad_feedback_token_len+[1]*len(bad_assistant_content_token_ids)+[0]*good_feedback_token_len+[1]*len(good_assistant_token_ids)
            else:
                user_token_ids = self.tokenizer.build_single_message("user","",dict_info["question"])
                assistant_token_ids = self.tokenizer.build_single_message("assistant","",dict_info["answer"])
                input_ids +=user_token_ids+assistant_token_ids+[self.tokenizer.eos_token_id]    
                target_mask+=[0]*len(user_token_ids)+[0]+[1]*(len(assistant_token_ids)-1)+[1]
            
        assert len(input_ids)==len(target_mask)
        if mask == "nomask":
            labels = [0,0]+[1]*(len(target_mask)-2)
        elif mask == "collapsed":
            labels = target_mask
        else:
            last_assistant_token_ids=self.tokenizer.build_single_message("assistant","",datadict['content'][-1]["answer"])
            last_message_len = len(last_assistant_token_ids)
            labels=[0]*len(target_mask)
            labels[-last_message_len:]=[1]*last_message_len
            
        input_ids = input_ids[:self.max_seq_len]
        target_mask = target_mask[:self.max_seq_len]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        return inputs
        
    def __build_qwen():
        pass
    
    def __build_llama():
        pass
        
    def __map_empathy(self,datadict,mask: str="collapsed",feedback_token:bool=True):
        '''
        mask:bool 用于控制是否屏蔽 人类 的输入，只计算 助手 的回答结果的损失
        datadict: {"conversation_id":"x","content":[{"question":"xxxx","answer":"xxxx"},
        {"question":"xxxx","answer":"xxxx"},...,
        {"question":"xxxx","answer":"xxxx"}]}
        '''
        inputs = self.__build_chatglm(datadict=datadict,mask=mask,feedback_token=feedback_token)
        return inputs


    def __map_mix_custom(self,datadict,mask: Union[str,"nomask","collapsed","ordinary"],feedback_token:bool=False):
        '''
        mask:bool 用于控制是否屏蔽 人类 的输入，只计算 助手 的回答结果的损失
        datadict: {"conversation_id":"x","content":[{"question":"xxxx","answer":"xxxx"},
        {"question":"xxxx","answer":"xxxx"},...,
        {"question":"xxxx","answer":"xxxx"}]}
        '''
        inputs = self.__build_chatglm(datadict=datadict,mask=mask,feedback_token=feedback_token)
        return inputs

    
    def __load_jsonlines(self,datapath):
        dataset=Dataset.from_generator(self.gen_from_jsonlines,gen_kwargs={"datapath":datapath})
        return dataset
    

    def generate_train_test_data(self,datapath,test_size,mask: Union[str,"nomask","collapsed","ordinary"],feedback_token:bool=False):
        data_name=datapath.split("/")[-2]
        if data_name.lower()=="mix_custom":
             logger.info("Loading Mix Custom multiturn-conversation.............")
             dataset=self.__load_jsonlines(datapath)
             map_func=self.__map_mix_custom
             column_names=dataset.column_names
        elif data_name.lower()=="empathy":
            logger.info("正在加载单轮对话共情数据.................")
            dataset=self.__load_jsonlines(datapath)
            map_func=self.__map_empathy
            column_names=dataset.column_names
        else:
            raise NotImplementedError
        
        if test_size>0:
            splitted_dataset=dataset.train_test_split(test_size=test_size,shuffle=True,seed=42) 
            train_dataset=splitted_dataset["train"].map(map_func,
                                                        fn_kwargs={"mask":mask,"feedback_token":feedback_token},
                                                        remove_columns=column_names,
                                                        num_proc=4)
            valid_dataset=splitted_dataset["test"].map(map_func,
                                                       fn_kwargs={"mask":mask,"feedback_token":feedback_token},
                                                       remove_columns=column_names,
                                                       num_proc=4)
        else:
            train_dataset=dataset.map(map_func,
                                      fn_kwargs={"mask":mask,"feedback_token":feedback_token},
                                      remove_columns=column_names,num_proc=1).shuffle()
            valid_dataset=None
        return train_dataset,valid_dataset
    

if __name__=="__main__":
    model_path="/data/models/chatglm3-6b-32k"
    tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    #tokenizer.padding_side="right"
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    datafile="data/sft_data/Mix_custom/CBT-modified.jsonl"
    #datafile="data/sft_data/empathy/empathy.jsonl"
    print("This is eos token"+tokenizer.eos_token)
    print("This is eos token id: "+str(tokenizer.eos_token_id))
    datagenerator=OpenSourceDataGen(tokenizer,512)
    train_dataset,cal_dataset=datagenerator.generate_train_test_data(datapath=datafile,
                                                                     test_size=0.001,
                                                                     mask="nomask",
                                                                     feedback_token=False)
    print(train_dataset[1]['input_ids'])
    print(tokenizer.decode(list(train_dataset[1]['input_ids'])))
    Dataloader=DL(train_dataset,batch_size=3,collate_fn=MultiturnConversationCollatorWithPadding(
        tokenizer=tokenizer,
        Max_seq_length=512
    ))
    for i,data in enumerate(Dataloader):
        if i==4:
            print(data["labels"][0])
            print(data["input_ids"].shape)
            break
