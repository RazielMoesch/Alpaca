import torch
import torch.nn as nn
from .embedding import Embedding
from .positional_encoding import PEncoding
from .encoder_block import EncoderBlock


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, ff_dim, num_layers, max_seq_len, device=None):
        super().__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding = Embedding(vocab_size, d_model).to(self.device)
        self.pos_encoding = PEncoding(d_model, max_seq_len).to(self.device)
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, ff_dim, device=self.device) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.embedding.forward(x)
        x += self.pos_encoding.forward(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)