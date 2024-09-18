import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.utils import PeftType
from peft.config import PeftConfig

@dataclass
class LilyConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lily`].

    Args:
        r (`int`): Lily's hidden dimension
        ne_1 (`int`): Lily's number of experts (ne) for lps
        ne_2 (`int`): Lily's number of experts (ne) for hps
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lily_s (`float`): The scaling factor for lily.
        lily_dropout (`float`): The dropout probability for Lily layers.
        modules_to_save (`List[str]`):List of modules apart from Lily layers to be set as trainable
            and saved in the final checkpoint.
        monoscale (`bool`): Whether to use the monoscale mode in Lily, set to false in default
    """

    r: int = field(default=4, metadata={"help": "Lily's hidden dimension"})
    ne_1: int = field(default=4, metadata={"help": "Lily's number of low-dimension projectors (lp)"})
    ne_2: int = field(default=4, metadata={"help": "Lily's number of high-dimension projectors (hp)"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lily."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lily_s: int = field(default=1.0, metadata={"help": "scaling factor for lily"})
    lily_dropout: float = field(default=0.0, metadata={"help": "Lily dropout"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from Lily layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers"
                " indexes that are specified inside this list. If a single integer is passed, PEFT will transform only"
                " the layer at this index."
            )
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer"
                " pattern is not in the common layers pattern."
            )
        },
    )
    rank_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 8`}"
            )
        },
    )
    init_weights: bool = field(
        default=False,
        metadata={
            "help": (
                "The initialization of the Lily weights. Right now not implemented."
            )
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.LILY
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")



