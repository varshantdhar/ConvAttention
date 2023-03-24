import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

class ScaledDotProductAttention(nn.Module):

    def __init__(self, dim) -> None:
        super().__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value, mask=None):
        att = torch.bmm()


class MultiHeadAttention(nn.Module):
     
    def __init__(self, n_embd=512, n_heads=8, bias=True, attn_drop=0.1, resid_drop=0.1, dropout=0.1) -> None:
        super().__init__()

        assert n_embd % n_heads == 0

        self.n_heads = n_heads
        self.n_embd = n_embd
        self.d_head = int(n_embd / n_heads)
        self.attn = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd) 

        self.attn_dropout = nn.Dropout(attn_drop)
        self.resid_dropout = nn.Dropout(resid_drop)
        self.dropout = dropout

        self.sqrt_dim = np.sqrt(n_embd)

        # scaled_dot_product_attention only available in PyTorch >= 2.0
        self.flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x, mask=None):
        B, T, C = x.size()

        q, k, v = self.attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2) # B, nh, T, dh
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2) # B, nh, T, dh
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2) # B, nh, T, dh

        if mask is not None:
            mask = mask.repeat(self.n_heads, 1, 1)
        

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout)




