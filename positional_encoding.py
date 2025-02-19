import torch
import math

class PEncoding():
    def __init__(self, embedding_dim, max_seq_len=128):
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        self.position_encodings = self.create_position_encodings(max_seq_len, embedding_dim)
    

    def create_position_encodings(self, max_seq_len, embedding_dim):
        position_encodings = torch.zeros(max_seq_len, embedding_dim)

        for pos in range(max_seq_len):
            for i in range(0, embedding_dim, 2):
                position_encodings[pos, i] = math.sin(pos/ (10000 **(i/embedding_dim)))

                if i + 1 < embedding_dim:
                    position_encodings[pos, i+1] = math.cos(pos/(10000**(i/embedding_dim)))
        
        return position_encodings

    def forward(self, input_ids):
        #print(f"pencoding forward pass input size: {input_ids.size()}")
        #input_ids = input_ids.squeeze(-1)
        #print(f"Pencoding input_ids post squeze: {input_ids.size()}")
        batch_size, seq_len= input_ids.size()[0], input_ids.size()[1]

        return self.position_encodings[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)


if __name__ == "__main__":

    voc_size = 10000
    embedding_dim = 512
    seq_len = 10

    pos_enc = PEncoding(voc_size, embedding_dim)

    input_ids = torch.randint(0, voc_size, (2,seq_len))

    position_encodings = pos_enc.forward(input_ids)

    print(position_encodings.shape)