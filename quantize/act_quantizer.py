import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import tqdm
import numpy as np
import pdb
import math
import time
CLIPMIN = 1e-5

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

class UniformActQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        group_size=None,
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits

        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.group_size = group_size
        self.enable = True

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1

    def fake_quant(self, x, scale, round_zero_point): 
        if self.group_size:
            dim1, dim2, dim3 = x.shape
            x = x.reshape(-1, self.group_size)
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2, dim3)
        return x_dequant
    

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        scale, round_zero_point = self.per_token_dynamic_calibration(x)

        x_dequant = self.fake_quant(x, scale, round_zero_point)
        return x_dequant

    def per_token_dynamic_calibration(self, x):
        if self.group_size:
            x = x.reshape(-1,self.group_size)

        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax =  x.amax(reduce_shape, keepdim=True)

        range = xmax - xmin
        scale = range / (2**self.n_bits-1)
        scale = scale.clamp(min=CLIPMIN, max=1e4)
        zero_point = -(xmin) / (scale)
        round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()

        return scale, round_zero_point