import torch
import struct
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy
from scipy.stats import norm
import numpy as np
import vtab
import yaml
import os
import random


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, output, label):
        self.sum += (output.argmax(dim=1).view(-1) == label.view(-1)).long().sum()
        self.count += label.size(0)

    def result(self):
        return self.sum / self.count
