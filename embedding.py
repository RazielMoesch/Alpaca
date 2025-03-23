import torch


class Embedding:
    def __init__(self, vocab_size, embedding_dim, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_matrix = torch.randn(vocab_size, embedding_dim, requires_grad=True).to(self.device)
    
    def forward(self, input_ids):
        input_ids = input_ids.type(torch.int64).to(self.device)
        return self.embedding_matrix[input_ids]


if __name__ == "__main__":
    vocab_size = 10
    embedding_dim = 5

    embedding_layer = Embedding(vocab_size, embedding_dim)

    input_ids = torch.tensor([1, 2, 3, 4])

    out = embedding_layer.forward(input_ids)
    print(out)