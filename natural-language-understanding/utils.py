import torch
from torch import nn

def set_trainbale_parameters(model):
    param = 0
    for n, p in model.named_parameters():
        if 'lily' in n or 'classifier' in n:
            p.requires_grad = True
            if 'lily' in n:
                # print(n)
                param += p.numel()
        else:
            p.requires_grad = False
    
    print(f"paramters is {param}")