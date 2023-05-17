import torch.nn as nn

from attention import MultiHeadAttention
from embeddings import PositionalEncoding
from sublayers import AddNorm, PositionWiseFFN

class TransformerEncoderLayer(nn.Module):
    """
    The Encoder layer is made up of self-attention and a feedforward netowrk.

    Args:
        d_model: dimension of model (default: 512)
        num_heads: number of attention heads (default: 8)
        d_ff: dimension of feed forward network (default: 2048)
        dropout_p: probability of dropout (default: 0.3)
        ffnet_style: style of feed forward network [ff, conv] (default: ff)
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, dropout_p: float = 0.3, ffnet_style: str = 'ff') -> None:
        super().__init__()
        self.self_attention = AddNorm(MultiHeadAttention(d_model, num_heads), d_model)
        self.feed_forward = AddNorm(PositionWiseFFN(d_model, d_ff, dropout_p, ffnet_style), d_model)

    def forward(self, inputs, self_attn_mask):
        output, attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        output = self.feed_forward(output)
        return output, attn


class TransformerEncoder(nn.Module):
    """
    The TransformerEncoder is composed of a stack of N identical encoder layers, where each encoder layer
    consists of a multi-headed self attention layer and a position wise fully connected feed forward layer

    Args:
        d_model: dimension of model (default: 512)
        input_dim: dimension of feature vector (default: 80)
        d_ff: dimension of feed forward network (default: 2048)
        num_layers: number of encoder layers (default: 6)
        num_heads: number of attention heads (default: 8)
        ffnet_style: style of feed forward networks [ff, conv] (default: ff)
        dropout_p: probability of dropout (default: 0.3)
        pad_id: identification of pad token (default: 0)
    """

    def __init__(self, d_model: int = 512, input_dim: int = 80, d_ff: int = 2048, num_layers: int = 6, num_heads: int =8, 
                ffnet_style: str = 'ff', dropout_p: float = 0.3, pad_id: int = 0) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers  = num_layers
        self.num_heads = num_heads
        self.pad_id = pad_id
        self.conv