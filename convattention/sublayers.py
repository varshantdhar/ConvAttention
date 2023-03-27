import torch.nn as nn

from convolution import conv1x1_block

class AddNorm(nn.Module):
    """
    Add & Normalization layer proposed in "Attention Is All You Need"
    for residual connection around each of the sub-layers
    """
    def __init__(self, sublayer, d_model=512) -> None:
        super().__init__()
        self.sublayer = sublayer
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, *args):
        residual = args[0]
        output = self.sublayer(*args)

        if isinstance(output, tuple):
            return self.layer_norm(output[0] + residual), output[1]
        
        return self.layer_norm(output + residual)

class PositionWiseFFN(nn.Module):
    """
    Position-wise Feedforward Networks are applied to each position separately 
    and identically. This consists of two linear transformations with a ReLU activation
    in between. Another way of describing this is as two convolutions with kernel size 1.
    """
    def __init__(self, d_model, d_ff, dropout_p, ffnet_style) -> None:
        super().__init__()
        self.ffnet_style = ffnet_style
        if ffnet_style == 'ffn':
            self.feed_forward = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_ff),
                nn.Dropout(dropout_p),
                nn.ReLU(),
                nn.Linear(in_features=d_ff, out_features=d_model),
                nn.Dropout(dropout_p)
            )
        
        elif ffnet_style == "conv":
            self.conv1 = conv1x1_block(in_channels=d_model, out_channels=d_ff)
            self.conv2 = nn.Conv2d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        
        else:
            raise ValueError("Unsupported Position-Wise FFN type. Pick between 'ffn' and 'conv'")

    
    def forward(self, x):
        if self.ffnet_style == 'conv':
            out = self.conv1(x.transpose(1, 2)) # (B, C, T) where C = nh * dh = d_model
            out = self.conv2(out).transpose(1, 2) # (B, T, C)
            return out
        
        return self.feed_forward(x)