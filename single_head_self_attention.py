import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleSelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.scale = embedding_dim ** .5
     
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.matmul(Q, K.transpose(-2, -1) / self.scale)

        attention_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, V)

        return output
    


if __name__ == "__main__":
    
    batch_size = 1
    seq_len = 5
    emb_dim = 10
    sa = SingleSelfAttention(emb_dim)
    in_vals = torch.randn(batch_size, seq_len, emb_dim)
    out = sa(in_vals)
    print(out)
    print(f"Shape: {out.shape}")

