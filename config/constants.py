"""Constant variables."""

from __future__ import annotations

__all__=[
    "SAFETY_PROMPT_FIELDS",
    "DEFAULT_BOS_TOKEN",
    "DEFAULT_EOS_TOKEN",
    "PROMPT_USER",
    "PROMPT_ASSISTANT",
    "PROMPT_INPUT",
    "PROMPT_DICT",
    "SEED",
    "LAYER_NORM_NAMES"
]

DEFAULT_BOS_TOKEN: str = "<s>"
DEFAULT_EOS_TOKEN: str = "</s>"
PROMPT_USER: str = "Human: {}"
PROMPT_ASSISTANT: str ="\n\nAssistant: {}"
PROMPT_INPUT=PROMPT_USER+PROMPT_ASSISTANT
SEED: int = 42
SAFETY_PROMPT_FIELDS: list[str] = [
                    'Unfairness_And_Discrimination', 
                    #'Crimes_And_Illegal_Activities', 
                    #'Insult', 
                    'Mental_Health',
                    #'Physical_Harm', 
                    'Privacy_And_Property',
                    'Ethics_And_Morality',
                    "all"]

PROMPT_DICT: dict[str,str] = {
    "prompt_user": PROMPT_USER,
    "prompt_assistant": PROMPT_ASSISTANT,
    "prompt_input": PROMPT_INPUT
}

LAYERNORM_NAMES = ["norm", "ln_f", "ln_attn", "ln_mlp"] # for LLaMA, BLOOM and Falcon settings