import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, d_model, ff_dim, device=None):
        super().__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.linear1 = nn.Linear(d_model, ff_dim).to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.linear2 = nn.Linear(ff_dim, d_model).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x 