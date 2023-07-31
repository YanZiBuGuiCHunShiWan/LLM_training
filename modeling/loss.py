from __future__ import annotations
import torch
import abc
import torch.nn as nn

__all__=[
    "Loss",
    "ShiftLabelMaskLoss",
    "ShiftWeightedLabelLoss",
    "PairwiseRMLoss",
    "RMMarginRankingLoss"
]

class Loss(object):
    """
    所有loss的类父类
    """
    @abc.abstractmethod
    def __call__(self, model, inputs, return_outputs=False):
        raise NotImplemented


class ShiftLabelMaskLoss(Loss):

    def __init__(self, ignore_index):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def __call__(self, model, inputs,return_outputs=False):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        target_mask = inputs['labels']
        #forward
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True) 
        #[Batch_size,seqlnegth,vocab_size]
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

        # 将labels中不属于target的部分，设为ignore_index，只计算target部分的loss
        labels = torch.where(target_mask == 1, input_ids, self.ignore_index)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
class ShiftWeightedLabelLoss(Loss):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.weighted_sequence=None

    def __call__(self, model, inputs,return_outputs=False):
        
       raise NotImplementedError
    
class PairwiseRMLoss(Loss):
    def __call__(self,model,inputs,return_outputs=False):
        batch_size=inputs["input_ids"].size(0)//2
        _,_,batch_scores=model(**inputs,output_hidden_states=True,return_dict=True) #batch_score:[2*batchsize,seqlength]
        better_scores,worse_scores=batch_scores.split(batch_size,dim=0)
        loss=-torch.log(torch.sigmoid(better_scores-worse_scores)).mean()
        return (loss,[loss,better_scores,worse_scores]) if return_outputs else loss
        

class RMMarginRankingLoss(Loss):
    def __init__(self,margin):
        super().__init__()
        self.marginloss=nn.MarginRankingLoss(margin)
    def __call__(self, model, inputs, return_outputs=False):
        batch_size=inputs.size(0)//2
        _,_,batch_scores=model(**inputs,output_hidden_states=True,return_dict=True) #batch_score:[2*batchsize,seqlength]
        better_scores,worse_scores=batch_scores.split(batch_size,dim=0)
        loss=self.marginloss(better_scores,worse_scores)
        return (loss,[loss,better_scores,worse_scores]) if return_outputs else loss
   