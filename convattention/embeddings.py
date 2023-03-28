import math
import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """
    "Attention is all you need" suggests using sine and cosine functions of different frequences:
        PE_(pos, 2i) = sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1) = cos(pos / power(10000, 2i / d_model))
    """
    def __init__(self, d_model=512, max_len=5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length):
        return self.pe[:, :length]