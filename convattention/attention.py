import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from convolution import conv3x3_dw_block, conv5x5_dw_block

class ScaledDotProductAttention(nn.Module):

    def __init__(self, dim, attn_drop) -> None:
        super().__init__()
        self.sqrt_dim = np.sqrt(dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, query, key, value, mask=None):
        # ((B * nh), T, dh) x ((B * nh), dh, T) -> ((B * nh), T, T)
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask == 0, -1e9)
        
        atten = self.attn_drop(F.softmax(score, dim=-1))
        context = torch.bmm(atten, value)
        return context, atten


class ConvAttentionPre(nn.Module):
    def __init__(self, channels, dim, attn_drop, use_bn = True, activation = "relu") -> None:
        super().__init__()
        self.conv = conv3x3_dw_block(channels=channels, use_bn=use_bn, activation=activation)
        self.sqrt_dim = np.sqrt(dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, query, key, value, mask=None):
        # ((B * nh), T, dh) x ((B * nh), dh, T) -> ((B * nh), T, T)
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        # Convolve the attention score before Softmax nnormalization
        conv_score = score + self.conv(score)

        if mask is not None:
            conv_score.masked_fill_(mask == 0, -1e9)

        # Calculate attention weights without convolution
        atten = self.attn_drop(F.softmax(conv_score, dim=-1))
         # Compute the attended output by matrix multiplication with value
        context = torch.bmm(atten, value)
        
        return context, atten

class ConvAttentionPost(nn.Module):

    def __init__(self, channels, dim, attn_drop, use_bn = True, activation = "relu") -> None:
        super().__init__()
        self.conv = conv3x3_dw_block(channels=channels, use_bn=use_bn, activation=activation)
        self.sqrt_dim = np.sqrt(dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, query, key, value, mask=None):
        # ((B * nh), T, dh) x ((B * nh), dh, T) -> ((B * nh), T, T)
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        if mask is not None:
            score.masked_fill_(mask == 0, -1e9)

        # Calculate attention weights without convolution
        atten = self.attn_drop(F.softmax(score, dim=-1))
        # Sum the attention weights and convolved attention weights
        conv_atten = atten + self.conv(atten)
         # Compute the attended output by matrix multiplication with value
        context = torch.bmm(conv_atten, value)

        return context, conv_atten


class MultiHeadAttention(nn.Module):
     
    def __init__(self, n_embd=512, n_heads=8, bias=True, attn_drop=0.1, resid_drop=0.1, dropout=0.1) -> None:
        super().__init__()

        assert n_embd % n_heads == 0

        self.n_heads = n_heads
        self.n_embd = n_embd
        self.d_head = int(n_embd / n_heads)
        self.attn = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd) 

        self.resid_dropout = nn.Dropout(resid_drop)
        self.dropout = dropout
        self.scaled_dot_prod_atten = ScaledDotProductAttention(self.d_head, attn_drop)

    def forward(self, query, key, value, mask=None):
        B = value.size(0)
        
        q = query.view(B, -1, self.n_heads, self.d_head) # (B, T, nh, dh)
        k = key.view(B, -1, self.n_heads, self.d_head) # (B, T, nh, dh)
        v = value.view(B, -1, self.n_heads, self.d_head) # (B, T, nh, dh)

        if mask is not None:
            mask = mask.repeat(self.n_heads, 1, 1)

        q = q.permute(2, 0, 1, 3).contiguous().view(B * self.n_heads, -1, self.d_head) # ((B * nh), T, dh)
        k = k.permute(2, 0, 1, 3).contiguous().view(B * self.n_heads, -1, self.d_head) # ((B * nh), T, dh)
        v = v.permute(2, 0, 1, 3).contiguous().view(B * self.n_heads, -1, self.d_head) # ((B * nh), T, dh)

        context, atten = self.scaled_dot_prod_atten.forward(query=q, key=k, value=v, mask=mask)
        
        context = context.view(self.n_heads, B, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(B, -1, self.n_heads * self.d_head) # (B, T, (nh * dh))

        return context, atten




        




