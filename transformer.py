import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .final_linear_layer import FinalLinear

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, ff_dim, num_layers, max_seq_len):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = Encoder(vocab_size, d_model, num_heads, ff_dim, num_layers, max_seq_len).to(self.device)
        self.decoder = Decoder(vocab_size, d_model, num_heads, ff_dim, num_layers, max_seq_len).to(self.device)
        self.final_linear = FinalLinear(d_model, vocab_size)
    
 
    def forward(self, src, tgt):

        src, tgt = src.to(self.device), tgt.to(self.device)
        encoder_out = self.encoder(src)
        decoder_out = self.decoder(tgt, encoder_out)
        return self.final_linear(decoder_out)