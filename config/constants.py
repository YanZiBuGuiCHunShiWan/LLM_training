"""Constant variables."""

from __future__ import annotations

__all__=[
    "SAFETY_PROMPT_FIELDS",
    "DEFAULT_BOS_TOKEN",
    "DEFAULT_EOS_TOKEN",
    "PROMPT_USER",
    "PROMPT_ASSISTANT",
    "PROMPT_INPUT",   
    "PROMPT_USER1",
    "PROMPT_ASSISTANT1",
    "PROMPT_INPUT1",
    "PROMPT_DICT",
    "PROMPT_DICT1",
    "SEED",
    "LAYER_NORM_NAMES"
]


################ prompt for custom format #################
DEFAULT_BOS_TOKEN: str = "<s>"
DEFAULT_EOS_TOKEN: str = "</s>"

PROMPT_USER_LLAMA: str = "Human: "
PROMPT_ASSISTANT_LLAMA: str ="Assistant: "

################ prompt for baichuan format ################
PROMPT_USER_BAICHUAN: str = "<reserved_106>"
PROMPT_ASSISTANT_BAICHUAN: str ="<reserved_107>"



################ prompt for qwen format ####################
PROMPT_USER_QWEN: str = ""
PROMPT_ASSISTANT_QWEN: str =""

FEEDBACK_TOKEN_DICT={
    "bad":"非提问式共情：",
    "good":"提问式共情："
}

SEED: int = 42
SAFETY_PROMPT_FIELDS: list[str] = [
                    'Unfairness_And_Discrimination', 
                    'Crimes_And_Illegal_Activities', 
                    'Insult', 
                    'Mental_Health',
                    'Physical_Harm', 
                    'Privacy_And_Property',
                    'Ethics_And_Morality',
                    "all",
                    FEEDBACK_TOKEN_DICT]
LAYERNORM_NAMES = ["norm", "ln_f", "ln_attn", "ln_mlp"] # for LLaMA, BLOOM and Falcon settings
