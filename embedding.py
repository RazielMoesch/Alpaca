import torch

'''
Explanation to understand:
    Create a matrix of vocab_size by embedding_dim .shape filled with random values. 
    return the value at the index of an input id
'''

class Embedding():
    def __init__(self, vocab_size, embedding_dim, device=None):
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_matrix = torch.randn(vocab_size, embedding_dim, requires_grad=True).to(device)
    
    def forward(self, input_ids):
        input_ids = input_ids.type(torch.int64)
        return self.embedding_matrix[input_ids]
    



if __name__ == "__main__":

    vocab_size =10
    embedding_dim = 5

    embedding_layer = Embedding(vocab_size, embedding_dim)


    input_ids = torch.tensor([1,2,3,4])

    out = embedding_layer.forward(input_ids)
    print(out)