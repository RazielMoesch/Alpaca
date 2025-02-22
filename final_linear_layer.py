import torch
import torch.nn as nn

class FinalLinear(nn.Module):
    
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        x = self.linear(x)
        #x = self.softmax(x)
        return x 