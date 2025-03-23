import torch
import torch.nn as nn


class FinalLinear(nn.Module):
    def __init__(self, d_model, vocab_size, device=None):
        super().__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.linear = nn.Linear(d_model, vocab_size).to(self.device)
        self.softmax = nn.Softmax(-1).to(self.device)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = x.to(self.device)
        x = self.linear(x)
        #x = self.softmax(x)
        return x