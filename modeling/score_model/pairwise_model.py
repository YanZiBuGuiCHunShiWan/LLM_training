import torch
import torch.nn as nn
from transformers import PreTrainedModel,Trainer
from peft import PeftModel
from trl import AutoModelForCausalLMWithValueHead
from typing import Dict,List,Optional,Literal


class PairwiseRewardModelTrainer(Trainer):
    def __init__(self,compute_loss: None,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.loss_func=compute_loss
        
    def compute_loss(
        self,
        model:PeftModel,
        inputs: Dict[str,torch.Tensor],
        return_outputs: Optional[bool] = False):
        '''
        How the loss is computed by the Pairwise Reward Model Trainer.
        '''
        return self.loss_func(model,inputs,return_outputs)

    