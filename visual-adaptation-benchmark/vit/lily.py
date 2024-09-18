import torch
from torch import nn
import torch.nn.functional as F
import timm
import matplotlib.pyplot as plt
import numpy as np

class lily_adapter(nn.Module):
    """
    """
    def __init__(self, in_dim, out_dim, hidden_dim, ne, lp, hps, mlp=False, idx=4):
        super().__init__()
        self.hps = hps
        self.ne = ne
        self.lp = lp
        self.router = nn.Linear(hidden_dim, ne, bias=False)
        if mlp:
            self.non_linear = nn.ReLU()
        else:
            self.non_linear = nn.Identity()
        self.idx = idx
    def forward(self, x):
        hidden = self.non_linear(self.lp(x))
        router_logits = self.router(hidden) # [B, N, num_of_experts]
        router_probability = F.softmax(router_logits, dim=-1) # [B, N, ne]
        expert_probabilities = router_probability.mean(dim=(0, 1)) 
        combined_hp = torch.einsum("e,eio->io", expert_probabilities, self.hps)
        return torch.matmul(hidden, combined_hp)
    
class lily_adapter_monoscale(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, ne, lp, hps, mlp=False):
        super().__init__()
        self.hps = hps
        self.ne = ne
        self.lp = lp
        self.scale = 1 / ne
        if mlp:
            self.non_linear = nn.ReLU()
        else:
            self.non_linear = nn.Identity()
    def forward(self, x):
        hidden = self.non_linear(self.lp(x))
        combined_hp = torch.sum(self.hps, 0) * self.scale
        return torch.matmul(hidden, combined_hp)

def forward_attn_lily_kv(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    delta_k = self.lily_k(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) * self.s
    delta_v = self.lily_v(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) * self.s
    q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
    k, v = k + delta_k[0], v + delta_v[0]
    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def forward_mlp_lily(self, x):
    delta_mlp = self.lily_mlp(x)
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop(x)
    x = self.fc2(x)
    x = self.drop(x) + (delta_mlp * self.s)
    return x



def set_lily_kv(model, dim=32, s=1, ne=4):
    print(f"Number of experts: {ne}")
    lily_k_hps = nn.Parameter(torch.zeros(ne, dim, 768))
    lily_v_hps = nn.Parameter(torch.zeros(ne, dim, 768))
    lily_mlp_hps = nn.Parameter(torch.zeros(ne, 2 * dim, 768))
    previous_k_lp = None
    previous_v_lp = None
    previous_mlp_lp = None
    idx = 0
    stride = 12 // ne
    for name, layer in model.named_modules():
        if type(layer) == timm.models.vision_transformer.Attention:
            if idx % stride == 0:
                print(f"set new lp at layer {idx}")
                previous_k_lp = nn.Linear(768, dim, bias=False)
                previous_v_lp = nn.Linear(768, dim, bias=False)

            layer.lily_k = lily_adapter(768, 768, dim, ne, previous_k_lp, lily_k_hps)
            layer.lily_v = lily_adapter(768, 768, dim, ne, previous_v_lp, lily_v_hps)
            layer.s = s
            bound_method = forward_attn_lily_kv.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
            
        elif type(layer) == timm.models.vision_transformer.Mlp:
            if idx % stride == 0:
                print(f"set new lp at layer {idx}")
                previous_mlp_lp = nn.Linear(768, 2 * dim, bias=False)

            layer.lily_mlp = lily_adapter(768, 768, 2 * dim, ne, previous_mlp_lp, lily_mlp_hps, mlp=True)
            layer.s = s
            bound_method = forward_mlp_lily.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
            idx += 1


def set_lily_kv_monoscale(model, dim=32, s=1, ne=4):
    print(f"Number of experts: {ne}")
    lily_k_hps = nn.Parameter(torch.zeros(ne, dim, 768))
    lily_v_hps = nn.Parameter(torch.zeros(ne, dim, 768))
    lily_mlp_hps = nn.Parameter(torch.zeros(ne, 2 * dim, 768))
    previous_k_lp = None
    previous_v_lp = None
    previous_mlp_lp = None
    idx = 0
    stride = 12 // ne
    for name, layer in model.named_modules():
        if type(layer) == timm.models.vision_transformer.Attention:
            if idx % stride == 0:
                print(f"set new lp at layer {idx}")
                previous_k_lp = nn.Linear(768, dim, bias=False)
                previous_v_lp = nn.Linear(768, dim, bias=False)

            layer.lily_k = lily_adapter_monoscale(768, 768, dim, ne, previous_k_lp, lily_k_hps)
            layer.lily_v = lily_adapter_monoscale(768, 768, dim, ne, previous_v_lp, lily_v_hps)
            layer.s = s
            bound_method = forward_attn_lily_kv.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
            
        elif type(layer) == timm.models.vision_transformer.Mlp:
            if idx % stride == 0:
                print(f"set new lp at layer {idx}")
                previous_mlp_lp = nn.Linear(768, 2 * dim, bias=False)

            layer.lily_mlp = lily_adapter_monoscale(768, 768, 2 * dim, ne, previous_mlp_lp, lily_mlp_hps, mlp=True)
            layer.s = s
            bound_method = forward_mlp_lily.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)
            idx += 1