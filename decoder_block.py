import torch
import torch.nn as nn
from .multi_head_self_attention import MultiSelfAttension
from .multi_head_cross_attention import MultiCrossAttention  
from .ffn import FFN


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim):
        super().__init__()  
        
        self.masked_attention = MultiSelfAttension(d_model, num_heads, masked=True)
        self.multi_cross = MultiCrossAttention(d_model, num_heads)  
        self.ffn = FFN(d_model, ff_dim)
        self.layer_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(0.1)
    

    def forward(self, x, encoder_output):
        masked_out = self.masked_attention(x)
        norm1 = self.layer_norm(x + self.dropout(masked_out))

        cross_out = self.multi_cross(norm1, encoder_output)
        norm2 = self.layer_norm(norm1 + self.dropout(cross_out))

        ffn_out = self.ffn(norm2)
        norm3 = self.layer_norm(norm2 + self.dropout(ffn_out))
        
        return norm3
