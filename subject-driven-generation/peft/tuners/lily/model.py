from __future__ import annotations

import math
import operator
import re
import warnings
from contextlib import contextmanager
from dataclasses import asdict, replace
from enum import Enum
from functools import partial, reduce
from itertools import chain
from typing import Literal, Optional, Dict

import torch
from torch import nn
from tqdm import tqdm

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
    onload_layer,
    replicate_layers,
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
    get_peft_model_state_dict,
    get_quantization_config,
)
from peft.utils.constants import TRANSFORMERS_MODELS_TO_LILY_TARGET_MODULES_MAPPING
from .config import LilyConfig
from .layer import LilyLayer, Linear

def _adapter_names_pre_forward_hook(target, args, kwargs, adapter_names):
    # pre-forward hook to inject the adapter_names argument when using mixed adapter batches inference
    kwargs["adapter_names"] = adapter_names
    return args, kwargs

class LilyModel(BaseTuner):
    """
    Creates Low-Rank Interconnectd Adaptation Across Layers (Lily) model from a pretrained transformers model.

    The method is described in detail in ???.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`LilyConfig`]): The configuration of the Lily model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The Lily model.

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LilyConfig`]): The configuration of the Lily model.
    """

    prefix: str = "lily_"

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)

    def _check_new_adapter_config(self, config: LilyConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        pass

    @staticmethod
    def _check_target_module_exists(lily_config, key):
        return check_target_module_exists(lily_config, key)

    def _create_and_replace(
        self,
        lily_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        lp,
        hp,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(lily_config.rank_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
        r = lily_config.rank_pattern.get(target_name_key, lily_config.r) # if there's any special rank setting
        lily_s = lily_config.lily_s
        ne_1 = lily_config.ne_1
        ne_2 = lily_config.ne_2
        out_features, in_features = target.weight.shape
        kwargs = {
            "in_features" : in_features,
            "out_features" : out_features,
        }

        if isinstance(target, LilyLayer):
            target.update_layer(
                adapter_name,
                r,
                lily_s=lily_s,
                lily_dropout=lily_config.lily_dropout,
                ne_2=ne_2
            )
        else:
            new_module = self._create_new_module(lily_config, adapter_name, target, lp, hp, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            if hasattr(new_module, "W_q"):  # HQQ
                new_module.W_q = child.W_q
            else:
                new_module.weight = child.weight

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if 'lily_' in name:
                module.to(child.weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

    @staticmethod
    def _create_new_module(lily_config, adapter_name, target, lp, hp, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        new_module = None
        new_module = Linear(target, adapter_name, r=lily_config.r, lily_s=lily_config.lily_s, lily_dropout=lily_config.lily_dropout, ne_1=lily_config.ne_1, ne_2=lily_config.ne_2, lp=lp, hp=hp, **kwargs)

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """Set the active adapter(s).

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for module in self.model.modules():
            if isinstance(module, LilyLayer):
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LILY_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LILY_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if hasattr(target, "base_layer"):
                self._replace_module(parent, target_name, target.get_base_layer(), target)
            elif isinstance(target, ModulesToSaveWrapper):
                # save any additional trainable modules part of `modules_to_save`
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if 'lily' not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LilyLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]

        self.active_adapter = new_adapter or []


    def unload(self) -> torch.nn.Module:
        """
        Gets back the base model by removing all the lora modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)

    def num_of_layers(self, model, adapter_name):
        peft_config = self.peft_config[adapter_name]
        one_target_key = None
        one_target_keys = list(peft_config.target_modules)
        counter = {}

        key_list = [key for key, _ in model.named_modules()]
        for one_target_key in one_target_keys:
            # target moduls found
            counter[one_target_key] = {}
            for key in key_list:
                if key.endswith(one_target_key):
                    # print(f"find {key} matching {one_target_key}")
                    parent, target, target_name = _get_submodules(model, key)
                    if target.weight.shape not in counter[one_target_key]:
                        counter[one_target_key][target.weight.shape] = 0
                    counter[one_target_key][target.weight.shape] += 1
        return counter
        
    def inject_adapter(self, model: nn.Module, adapter_name: str, autocast_adapter_dtype: bool = True) -> None:
        """
        Override BaseTuner to allow custom deployment of adapters in Lily.
        """
        peft_config = self.peft_config[adapter_name]
        self._check_new_adapter_config(peft_config)

        _check_for_modules_to_save = getattr(peft_config, "modules_to_save", None) is not None
        _has_modules_to_save = False
        model_config = getattr(model, "config", {"model_type": "custom"})
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()
        peft_config = self._prepare_adapter_config(peft_config, model_config)
        self._prepare_model(peft_config, model)
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in model.named_modules()]
        num_of_target: int = len(peft_config.target_modules)
        counter: int = 0
        lps: Dict[str, Dict] = {}
        hps: Dict[str, Dict] = {}
        for key in peft_config.target_modules:
            # initialize all the lps and hps
            lps[key] = {}
            hps[key] = {} # store differnt weight types
        num_layers = self.num_of_layers(model, adapter_name)
        stride = {}
        for target in num_layers.keys():
            stride[target] = {}
            for shape in num_layers[target].keys():
                stride[target][shape] = num_layers[target][shape] // peft_config.ne_1
        print(f"stride is {stride}")
        print(f"targets: {num_of_target}")
        print(f"target: {peft_config.target_modules}")
        print(f"ne 1 : {peft_config.ne_1}")
        print(f"ne 2 : {peft_config.ne_2}")
        counter = {}
        for key in peft_config.target_modules:
            counter[key] = {}
        idx = 0 # index into diferent parts of the Unet model
        for key in key_list:
            # find target modules
            if isinstance(peft_config.target_modules, str):
                target_module_found = re.fullmatch(peft_config.target_modules, key)
                matched_target = peft_config.target_modules if target_module_found else None
            else:
                for target_key in peft_config.target_modules:
                    if key.endswith(target_key):
                        target_module_found = True
                        matched_target = target_key
                        break
                else:
                    target_module_found = False
                    matched_target = None
            # target moduls found
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(model, key)

                self.targeted_module_names.append(key)
                if isinstance(target, torch.nn.Linear):
                    out_features, in_features = target.weight.shape
                    shape = target.weight.shape

                    if shape not in hps[matched_target]:
                        hps[matched_target][shape] = nn.Linear(out_features, peft_config.ne_2 * peft_config.r, bias=False)
                    if shape not in counter[matched_target]:
                        counter[matched_target][shape] = 0

                    if counter[matched_target][shape] % stride[matched_target][shape] == 0:
                        # setting new lp across 'stride' layers.
                        # print(f"setting new lps at {counter[matched_target][shape]} with stride {stride} with matched {matched_target}")

                        lps[matched_target][shape] = nn.Linear(in_features=in_features, out_features=peft_config.r, bias=False)
                        # print(lps[matched_target] == None)
                    self._create_and_replace(peft_config, adapter_name, target, target_name, parent, current_key=key, lp=lps[matched_target][shape], hp=hps[matched_target][shape])
                counter[matched_target][shape] += 1

        self.set_adapter(self.active_adapters)
        self._mark_only_adapters_as_trainable(model)
