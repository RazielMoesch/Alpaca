from .tokenizer import Tokenizer  
from .embedding import Embedding
from .positional_encoding import PEncoding
from .ffn import FFN
from .multi_head_self_attention import MultiSelfAttension
from .encoder_block import EncoderBlock
from .multi_head_cross_attention import MultiCrossAttention
from .decoder_block import DecoderBlock
from .final_linear_layer import FinalLinear
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .transformer import Transformer
from .train import train
from .alpaca_dataset import AlpacaDataset
from .validate import validate


class Alpaca:
    def __init__(self):
        self.tokenizer = Tokenizer()

    def token_embedding(self, vocab_size, embedding_dim):
        Embedding_layer = Embedding(vocab_size, embedding_dim)
        return Embedding_layer
    
    def pos_encoding(self, embedding_dim, max_seq_len):
        pencoding = PEncoding(embedding_dim, max_seq_len)
        return pencoding
    
    def ffn(self, d_model, ff_dim):
        ffnetwork = FFN(d_model, ff_dim)
        return ffnetwork
    
    def multi_self_attention(self, d_model, num_heads, masked=False):
        msa = MultiSelfAttension(d_model, num_heads, masked)
        return msa
    
    def encoder_block(self, d_model, num_heads, ff_dim):
        encoder = EncoderBlock(d_model, num_heads, ff_dim)
        return encoder
    
    def multi_cross_attention(self, d_model, num_heads):
        mca = MultiCrossAttention(d_model, num_heads)
        return mca
    
    def decoder_block(self, d_model, num_heads, ff_dim):
        decoder = DecoderBlock(d_model, num_heads, ff_dim)
        return decoder
    
    def final_linear_layer(self, d_model, vocab_size):
        lin = FinalLinear(d_model, vocab_size)
        return lin
    
    def transformer(self, vocab_size=5000, d_model=512, num_heads=8, ff_dim=2048, num_layers=6, max_seq_len=512):
        self.transformer = Transformer(vocab_size, d_model, num_heads, ff_dim, num_layers, max_seq_len)
        return self.transformer
    
    def new_transformer(self, vocab_size, d_model, num_heads, ff_dim, num_layers, max_seq_len):
        self.transformer = Transformer(vocab_size, d_model, num_heads, ff_dim, num_layers, max_seq_len)
        return self.transformer
    
    def dataset(self, txt_file, tokenizer=None, max_seq_len=512, merges=5000):
        if not tokenizer:
            tokenizer = self.tokenizer
        return AlpacaDataset(txt_file, tokenizer, max_seq_len, num_merges=merges)
    
    def train_model(self, epochs, train_dl, optimizer=torch.optim.Adam, transformer=None, loss_fn=nn.CrossEntropyLoss, lr=1e-4, validate_data=False, validation_data=None, wandb_tracking=False, lr_scheduler=False):
        if not transformer:
            transformer = self.transformer
        train(epochs, transformer, loss_fn, train_dl, optimizer, validate_data=validate_data, validation_dl=validation_data, wandb_tracking=wandb_tracking, lr_scheduler=lr_scheduler)
    
    def validate_model(self, model, val_dl, device):
        if not model:
            model = self.transformer
        validate(model, val_dl, device)

    
    def inference(self, tokens, detokenize=False):
        transformer = self.transformer

        tokens = tokens.unsqueeze(0)  

        transformer.eval()  
        with torch.inference_mode(): 
            output = transformer(tokens, tokens)  
        
        
        output = output.argmax(dim=-1) 

        if detokenize:
            result = [token.item() for token in output.squeeze(0)]
            
            detokenized_result = alpaca.tokenizer.detokenize(result)

            return detokenized_result
        
        return output




    



    
        
    

    

        
       

if __name__ == "__main__":
    
    alpaca = Alpaca()
    transformer = alpaca.transformer(vocab_size=100,
                                    d_model=8,
                                    num_heads=8,
                                    num_layers=2,
                                    max_seq_len=64)
    tokenizer = alpaca.tokenizer
    train_dataset = alpaca.dataset('example.txt', tokenizer, max_seq_len=64)
    print(f"Train dataset len: {len(train_dataset)}")
    print(f"Label min: {train_dataset[0].min()}, Label max: {train_dataset[0].max()}")

    print(f"train_dataset[1]: {train_dataset[1]}")
    
    train_dl = DataLoader(train_dataset, batch_size=2, shuffle=True)


    EPOCHS = 10000
    optimizer = torch.optim.Adam
    loss_fn = nn.CrossEntropyLoss
    LR = 1e-4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transformer.to(device)

    alpaca.train_model(EPOCHS, train_dl, optimizer, transformer=transformer, loss_fn=loss_fn, lr=LR)

    result = alpaca.inference(train_dataset[0], detokenize=True)
    print(result)
    

   
    
    
    