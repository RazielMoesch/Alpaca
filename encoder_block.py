import torch
import torch.nn as nn
from .multi_head_self_attention import MultiSelfAttension
from .ffn import FFN


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, device=None):
        super().__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.attention = MultiSelfAttension(d_model, num_heads).to(self.device)
        self.ffn = FFN(d_model, ff_dim).to(self.device)
        self.layer_norm1 = nn.LayerNorm(d_model).to(self.device)
        self.layer_norm2 = nn.LayerNorm(d_model).to(self.device)
        self.dropout = nn.Dropout(0.1).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        attn_output = self.attention(x)
        attn_output = self.layer_norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(attn_output)
        output = self.layer_norm2(attn_output + self.dropout(ffn_output))
        return output