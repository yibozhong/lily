import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from peft.tuners.tuners_utils import BaseTunerLayer
import einops
import random

class LilyLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lily_lp", "lily_hp", "lily_router")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lily_s", "lily_dropout", "ne_1", "ne_2")
    def __init__(
        self,
        base_layer: nn.Module, 
        ephemeral_gpu_offload: bool = False,
        **kwargs
    ) -> None:
        self.base_layer = base_layer
        self.r = 4
        self.lily_s = 1.0
        self.ne_1 = 4
        self.ne_2 = 4
        self.lily_lp = None
        self.lily_hp = None
        self.lily_router = None
        # Optional dropout
        self.lily_dropout = None
        self.kwargs = kwargs
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise NotImplementedError("only support layers of type \'nn.Linear\'")
    
    def update_layer(
        self, adapter_name, r, lily_s, lily_dropout, lp, hp, ne_2
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r = r
        if lily_dropout > 0:
            lily_dropout_layer = nn.Dropout(p=lily_dropout)
        else:
            lily_dropout_layer = nn.Identity()
        self.lily_dropout = lily_dropout_layer
        # actual trainble parameters
        self.lily_lp = lp
        self.lily_hp = hp
        self.lily_router = nn.Linear(r, ne_2, bias=False)
        self.lily_s = lily_s
        self.ne_2 = ne_2

    def reset_lily_parameters(self, adapter_name):
        if adapter_name in self.lily_lp.keys():
            nn.init.kaiming_uniform_(self.lily_lp.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lily_router.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lily_hp.weight)
    
    def set_adapter(self, adapter_name):
        # disregard the adaptername parameter for now.
        self.lily_lp.requires_grad_(True)
        self.lily_hp.requires_grad_(True)
        self.lily_router.requires_grad_(True)
    



class Linear(nn.Module, LilyLayer):
    # Lily implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lily_s: float = 1.0,
        lily_dropout: float = 0.0,
        ne_1: int = 4,
        ne_2: int = 4,
        lp: nn.Linear = None,
        hp: nn.Linear = None,
        **kwargs,
    ) -> None:
        super().__init__()
        LilyLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lily_s=lily_s,
            lily_dropout=lily_dropout,
            lp=lp,
            hp=hp,
            ne_2=ne_2,
        )
    def get_delta_weight(self, adapter, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lily_hp.device
        dtype = self.lily_hp.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        lp = self.lily_lp.weight
        hp = self.lily_hp.weight
        router = self.lily_router.weight

        if cast_to_fp32:
            lp = lp.float()
            hp = hp.float()
        
        hidden = self.lily_lp(x)
        router_logits = torch.matmul(hidden, router.t())
        router_probability = F.softmax(router_logits, dim=-1) # [B, N, ne]
        expert_probabilities = router_probability.mean(dim=(0, 1)) 
        hp = einops.rearrange(hp, "(e i) o -> e i o", e=self.ne_2)
        combined_hp = torch.einsum("e,eio->io", expert_probabilities, hp)
        output_tensor = torch.matmul(lp.t(), combined_hp) * self.lily_s # get the approximated weight update

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lily_lp.weight.data = lp.to(dtype)
            self.lily_hp.weight.data = hp.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # if random.random() < 0.0001:
        #     print("lily forward")
        # else:
        #     pass
        adapter_names = kwargs.pop("adapter_names", None)

        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        lily_lp = self.lily_lp
        lily_hp = self.lily_hp
        hp = einops.rearrange(lily_hp.weight, "(e i) o -> e i o", e=self.ne_2)
        dropout = self.lily_dropout
        scaling = self.lily_s
        x = x.to(lily_lp.weight.dtype)
        hidden = self.lily_lp(x)
        router_logits = self.lily_router(hidden) # [B, N, num_of_experts]
        router_probability = F.softmax(router_logits, dim=-1) # [B, N, ne]
        expert_probabilities = router_probability.mean(dim=(0, 1)) 
        combined_hp = torch.einsum("e,eio->io", expert_probabilities, hp)
        delta = torch.matmul(hidden, combined_hp)
        result = result + (delta * scaling)

        result = result.to(torch_result_dtype)

        return result