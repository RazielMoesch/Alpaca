import torch
import torch.nn as nn
from .embedding import Embedding
from .positional_encoding import PEncoding
from .encoder_block import EncoderBlock


class Encoder(nn.Module): 
    def __init__(self, vocab_size, d_model, num_heads, ff_dim, num_layers, max_seq_len, device=None):
        super().__init__()

        self.device = device
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = PEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, ff_dim).to(device) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model).to(device)

    
    def forward(self, x):
        if not self.device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x = self.embedding.forward(x) 
        x = x.unsqueeze(0) if x.dim() == 1 else x  
        x = x.cpu()
        x += self.pos_encoding.forward(x)
        x = x.to(device)
        for layer in self.layers:
            x = layer(x)
        
        return self.norm(x)