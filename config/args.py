from __future__ import annotations

from dataclasses import field, fields, dataclass
from typing import Optional,Literal

__all__=[
    "FinetuneArguments"
]

@dataclass
class FinetuneArguments:
    '''
    微调参数设置
    '''
    #通用参数设置   
    
    model_name_or_path: str = field()
    
    model_type: Optional[Literal["baichuan","chatglm"]] = field(
        default="baichuan",
        metadata={"help":"Type of your model. set 'baichuan' for 'LLama,Bloom,Baichuan'"}
    )
    
    train_file: str = field(
        default="data/sft_data/Safetyprompts/typical_safety_scenarios.json"
        )
    
    test_size: float = field(
        default=0.2
        )
    
    quantization: Optional[Literal["4bit","8bit",None]] = field(
        default=None,
        metadata={"help":"  "})
    
    max_seq_length: int = field(
        default=1024,
        metadata={"help":"The maximum length of the sequence"}
        )
    
    finetuning_type: Optional[Literal["lora","full"]]= field(
        default="lora",
        metadata={"help":"Only 'lora' and full-parameter tuning are supported currently."}
    )
    
    training_stage: Optional[Literal["sft","rm"]] = field(
        default="sft",
        metadata={"help":"ONly 'sft' and 'rm' are supported currently."}
    )
    #LoRA参数设置
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help":"lora rank"}
        )
    
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help":"lora dropout"}
        )
    
    lora_alpha: Optional[int] = field(
        default=16,
        metadata={"help": "lora alpha"})
    