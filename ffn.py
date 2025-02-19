import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, d_model, ff_dim):
        super().__init__()


        self.linear1 = nn.Linear(d_model, ff_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ff_dim, d_model)

    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


