import torch
import torch.nn as nn
from .multi_head_self_attention import MultiSelfAttension
from .multi_head_cross_attention import MultiCrossAttention
from .ffn import FFN


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, device=None):
        super().__init__()

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.masked_attention = MultiSelfAttension(d_model, num_heads, masked=True).to(self.device)
        self.multi_cross = MultiCrossAttention(d_model, num_heads).to(self.device)
        self.ffn = FFN(d_model, ff_dim).to(self.device)
        self.layer_norm = nn.LayerNorm(d_model).to(self.device)
        self.dropout = nn.Dropout(0.1).to(self.device)

    def forward(self, x, encoder_output):
        x = x.to(self.device)
        encoder_output = encoder_output.to(self.device)

        masked_out = self.masked_attention(x)
        norm1 = self.layer_norm(x + self.dropout(masked_out))

        cross_out = self.multi_cross(norm1, encoder_output)
        norm2 = self.layer_norm(norm1 + self.dropout(cross_out))

        ffn_out = self.ffn(norm2)
        norm3 = self.layer_norm(norm2 + self.dropout(ffn_out))

        return norm3