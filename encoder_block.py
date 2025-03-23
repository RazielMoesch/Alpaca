import torch
import torch.nn as nn
from .multi_head_self_attention import MultiSelfAttension
from .ffn import FFN


class EncoderBlock(nn.Module): 
    def __init__(self, d_model, num_heads, ff_dim): 
        super().__init__()

        self.attention = MultiSelfAttension(d_model, num_heads)

        self.ffn = FFN(d_model, ff_dim)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):

        attn_output = self.attention(x) 
        attn_output = self.layer_norm1(x + self.dropout(attn_output)) 

        
        ffn_output = self.ffn(attn_output)  
        output = self.layer_norm2(attn_output + self.dropout(ffn_output)) 

        return output
