import torch,os
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    TrainingArguments,
    DataCollator,
    PreTrainedTokenizerBase,
    EvalPrediction,
    TrainerCallback,
    Trainer
)
from peft import PeftModel
from transformers.utils import (
    logging
)
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import Dataset
# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

logger = logging.get_logger(__name__)

class SFTTrainer(Trainer):
    def __init__(self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
            compute_loss=None,):
        
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.loss_func=compute_loss
        
    def compute_loss(
        self,
        model: PeftModel,
        inputs: Dict[str,torch.Tensor],
        return_outputs: Optional[bool] = False):
        '''
        
        '''
        if self.loss_func is None:
            loss = super().compute_loss(model, inputs, return_outputs)
        else:
            loss = self.loss_func(model, inputs, self.args, return_outputs)
        return loss
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # 保存lora权重和配置
        self.model.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
        )
        
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        