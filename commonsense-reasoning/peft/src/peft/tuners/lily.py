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

from ..utils import PeftConfig, PeftType, transpose

def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


if is_bnb_available():
    import bitsandbytes as bnb

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

    r: int = field(default=8, metadata={"help": "Lily's hidden dimension"})
    ne_1: int = field(default=2, metadata={"help": "Lily's number of low-dimension projectors (lp)"})
    ne_2: int = field(default=2, metadata={"help": "Lily's number of high-dimension projectors (hp)"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lily."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lily_s: int = field(default=None, metadata={"help": "scaling factor for lily"})
    lily_dropout: float = field(default=None, metadata={"help": "Lily dropout"})
    monoscale: bool = field(default=False, metadata={"help": "set the mode in Lily as monoscale with no router"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from Lily layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.LILY


def mark_only_lily_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if "lily_" not in n:
            p.requires_grad = False
        else:
            p.requires_grad = True
    

class LilyModel(torch.nn.Module):
    """
    Creates Low-Rank Interconnected Adapter across layers (Lily) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LilyConfig`]): The configuration of the Lily model.

    Returns:
        `torch.nn.Module`: The Lily model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LilyConfig >>> from peft import LilyModel, LilyConfig >>>
        config = LilyConfig(
            peft_type="LILY", task_type="SEQ_2_SEQ_LM", r=8, lily_s=32, target_modules=["q", "v"],
            lily_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lily_model = LilyModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LilyConfig`]): The configuration of the Lily model.
    """

    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_lily_as_trainable(self.model)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.peft_config.r,
            "lily_s": self.peft_config.lily_s,
            "lily_dropout": self.peft_config.lily_dropout,
        }

        # set the lps and hps
        num_of_target: int = len(self.peft_config.target_modules)
        counter: int = 0
        lps: Dict[str, nn.Linear] = {}
        hps: Dict[str, nn.Parameter] = {}
        for key in self.peft_config.target_modules:
            # initialize all the lps and hps
            lps[key] = None
            hps[key] = None

        stride = (self.model.config.num_hidden_layers // self.peft_config.ne_1)
        print(f"targets: {num_of_target}")
        print(f"target: {self.peft_config.target_modules}")
        print(f"ne 1 : {self.peft_config.ne_1}")
        print(f"ne 2 : {self.peft_config.ne_2}")
        print(f"number of layers: {self.model.config.num_hidden_layers}")

        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:

            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
                matched_target = self.peft_config.target_modules if target_module_found else None
            else:
                for target_key in self.peft_config.target_modules:
                    if key.endswith(target_key):
                        target_module_found = True
                        matched_target = target_key
                        break
                else:
                    target_module_found = False
                    matched_target = None

            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None

                if isinstance(target, torch.nn.Linear):
                    # print(f"matched {matched_target} with counter {counter}")
                    out_features, in_features = target.weight.shape

                    if hps[matched_target] == None:
                        hps[matched_target] = nn.Parameter(torch.zeros(self.peft_config.ne_2, self.peft_config.r, out_features))

                    if (counter // num_of_target) % stride == 0:
                        # setting new lp across 'stride' layers.
                        print(f"setting new lps at {counter} with stride {stride} with matched {matched_target}")
                        lps[matched_target] = nn.Linear(in_features=in_features, out_features=self.peft_config.r, bias=False)
                        # print(lps[matched_target] == None)

                    if not self.peft_config.monoscale:
                        new_module = Linear(in_features, out_features, r=self.peft_config.r, lily_s=self.peft_config.lily_s,
                                            lily_dropout=self.peft_config.lily_dropout, ne=self.peft_config.ne_2, lp=lps[matched_target],
                                            hp=hps[matched_target]
                                        )
                    else:
                        new_module = Linear_mono(in_features, out_features, r=self.peft_config.r, lily_s=self.peft_config.lily_s,
                                            lily_dropout=self.peft_config.lily_dropout, ne=self.peft_config.ne_2, lp=lps[matched_target],
                                            hp=hps[matched_target]
                                        )
                counter += 1

                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lily_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LilyLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


class LilyLayer:
    def __init__(
        self,
        r: int,
        lily_s: float,
        lily_dropout: float,
        ne: int,
    ):
        self.r = r
        self.lily_s = lily_s
        self.ne = ne
        # Optional dropout
        if lily_dropout > 0.0:
            self.lily_dropout = nn.Dropout(p=lily_dropout)
        else:
            self.lily_dropout = lambda x: x
        self.disable_adapters = False

class Linear(nn.Linear, LilyLayer):
    # Lily implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lily_s: float = 1.0,
        lily_dropout: float = 0.0,
        ne: int = 4,
        lp: nn.Linear = None,
        hp: nn.Parameter = None,
    ):
        nn.Linear.__init__(self, in_features, out_features)
        LilyLayer.__init__(self, r=r, lily_s=lily_s, lily_dropout=lily_dropout, ne=ne)
        if r == 0:
            raise NotImplementedError
        self.lily_hp = hp
        self.lily_lp = lp
        # print(f"lp is None {self.lily_lp == None}")
        # print(f"hp is None {self.lily_hp == None}")
        self.lily_router = nn.Linear(r, ne, bias=False)
        self.weight.requires_grad = False
        self.reset_parameters()
        # if mlp:
        #     self.non_linear = nn.ReLU()
        # else:
        self.non_linear = nn.Identity()
        # self.idx = idx
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lily_lp"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lily_lp.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lily_router.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lily_hp)
    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lily_lp.train(mode)
        # self.lily_hp.train(mode)
        self.lily_router.train(mode)

    def eval(self):
        nn.Linear.eval(self)
        self.lily_lp.eval()
        # self.lily_hp.eval()
        self.lily_router.eval()

    def forward(self, x: torch.Tensor):
        previous_dtype = self.weight.dtype
        current_dtype = self.lily_lp.weight.dtype
        base = torch.matmul(x, self.weight.t())
        hidden = self.non_linear(self.lily_lp(x.to(current_dtype)))
        router_logits = self.lily_router(hidden) # [B, N, num_of_experts]
        router_probability = F.softmax(router_logits, dim=-1) # [B, N, ne]
        expert_probabilities = router_probability.mean(dim=(0, 1)) 
        combined_hp = torch.einsum("e,eio->io", expert_probabilities, self.lily_hp)
        result = torch.matmul(hidden, combined_hp)
        result = result + base
        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result


class Linear_mono(nn.Linear, LilyLayer):
    # Lily implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lily_s: float = 1.0,
        lily_dropout: float = 0.0,
        ne: int = 4,
        lp: nn.Linear = None,
        hp: nn.Parameter = None,
    ):
        nn.Linear.__init__(self, in_features, out_features)
        LilyLayer.__init__(self, r=r, lily_s=lily_s, lily_dropout=lily_dropout, ne=ne)
        if r == 0:
            raise NotImplementedError
        self.lily_hp = hp
        self.lily_lp = lp
        self.weight.requires_grad = False
        self.reset_parameters()
        # if mlp:
        #     self.non_linear = nn.ReLU()
        # else:
        self.non_linear = nn.Identity()
        self.scale = 1 / self.ne
        # self.idx = idx
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lily_lp"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lily_lp.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lily_hp)
    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lily_lp.train(mode)
        # self.lily_hp.train(mode)

    def eval(self):
        nn.Linear.eval(self)
        self.lily_lp.eval()
        # self.lily_hp.eval()

    def forward(self, x: torch.Tensor):
        previous_dtype = self.weight.dtype
        current_dtype = self.lily_lp.weight.dtype
        print(f"previous_dtype {previous_dtype}")
        print(f"current_dtype {current_dtype}")
        base = torch.matmul(x, self.weight.t())
        hidden = self.non_linear(self.lily_lp(x.to(current_dtype)))
        combined_hp = torch.sum(self.lily_hp, 0) * self.scale
        result = torch.matmul(hidden, combined_hp)
        result = result + base
        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result
    

if is_bnb_available():

    class Linear8bitLt(bnb.nn.Linear8bitLt, LilyLayer):
        # Lily implemented in a dense layer with 8-bit quantization
        def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lily_s: float = 1.0,
            lily_dropout: float = 0.0,
            ne: int = 4,
            lp: nn.Linear = None,
            hp: nn.Parameter = None,
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            LilyLayer.__init__(self, r=r, lily_s=lily_s, lily_dropout=lily_dropout, ne=ne)
            if r == 0:
                raise NotImplementedError
            self.lily_hp = hp
            self.lily_lp = lp
            self.lily_router = nn.Linear(r, ne, bias=False)
            self.weight.requires_grad = False
            self.reset_parameters()
            self.non_linear = nn.Identity()

        def reset_parameters(self):
            if hasattr(self, "lily_lp"):
                nn.init.kaiming_uniform_(self.lily_lp.weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.lily_router.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lily_hp)

        def train(self, mode: bool = True):
            bnb.nn.Linear8bitLt.train(self, mode)
            self.lily_lp.train(mode)
            self.lily_router.train(mode)

        def eval(self):
            bnb.nn.Linear8bitLt.eval(self)
            self.lily_lp.eval()
            self.lily_router.eval()

        def forward(self, x: torch.Tensor):
            previous_dtype = self.weight.dtype
            current_dtype = self.lily_lp.weight.dtype
            result = super().forward(x).to(current_dtype)
            hidden = self.non_linear(self.lily_lp(x.to(current_dtype)))
            router_logits = self.lily_router(hidden)  # [B, N, num_of_experts]
            router_probability = F.softmax(router_logits, dim=-1)  # [B, N, ne]
            expert_probabilities = router_probability.mean(dim=(0, 1))
            combined_hp = torch.einsum("e,eio->io", expert_probabilities, self.lily_hp)
            result += torch.matmul(hidden, combined_hp)
            if result.dtype != previous_dtype:
                result = result.to(previous_dtype)

            return result