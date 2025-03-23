```markdown
# This File will go through most of the methods directly accessible through the Alpaca class.
# Note: This md file was made with help from deepseekAI's chatbot to help with grammar, wording, and formatting since its better than me at it.

## Table of Contents

1. [Core-Transformer-Components](#Core-Transformer-Components)
    - [Layout](#Layout)
    - [Encoder](#Encoder)
        - [Embedding](#Embedding)
        - [Positional-Encoding](#Positional-Encoding)
        - [Encoder-Blocks](#Encoder-Blocks)
            - [Multi-Head-Self-Attention](#Multi-Head-Self-Attention)
            - [Feed-Forward-Network](#Feed-Forward-Network)
    - [Decoder](#Decoder)
        - [Embedding](#Embedding)
        - [Positional-Encoding](#Positional-Encoding)
        - [Decoder-Blocks](#Decoder-Blocks)
            - [Multi-Head-Self-Attention](#Multi-Head-Self-Attention)
            - [Multi-Head-Cross-Attention](#Mutli-Cross-Self-Attention)
            - [Feed-Forward-Network](#Feed-Forward-Network)

2. [Transformer-Implementation](#Transformer-Implementation)
    - [Creating-an-Alpaca-Transformer](#Creating-a-Transformer)
    - [Creating-an-Alpaca-Dataset](#Creating-an-Alpaca-Dataset)
    - [Training-an-Alpaca-Transformer](#Training-an-Alpaca-Transformer)
    - [Creating-Predictions-Using-an-Alpaca-Transformer](#Creating-Predictions-Using-an-Alpaca-Transformer)

---

# Core Transformer Components

This Section will go over the core functionality and computation going on behind the scenes as well as the specific functions and methods are accessible through an 'Alpaca' object.

---

## Layout

The layout of an Alpaca-Transformer is according to the following Structure:

---

### Encoder

---

#### Embedding

The Embedding Layer is accessed through the `Alpaca.token_embedding()` method. The method takes in two params: `vocab_size` and `embedding_dim`. In summary, the embedding layer is a trainable lookup table that maps discrete token IDs (integers) to continuous vector representations. It initializes as a matrix of random values with dimensions `(vocab_size x embedding_dim)`, where each row corresponds to a token's embedding. During the forward pass, it retrieves the embedding vectors for the input token IDs, enabling the model to process text as dense, meaningful vectors. These embeddings are optimized during training to capture semantic and syntactic relationships between tokens.

---

##### Here is a code-based demonstration of the embedding layer concept:

```python
import torch

# Define the embedding layer
vocab_size = 4  # Number of unique tokens in the vocabulary
embedding_dim = 4  # Dimensionality of the embedding vectors

# Randomly initialize the embedding matrix
embedding_matrix = torch.randn(vocab_size, embedding_dim, requires_grad=True)

# Example input: Token IDs
input_ids = torch.tensor([1, 2, 3, 0])  # Shape: (sequence_length,)

# Forward pass: Retrieve embeddings
output = embedding_matrix[input_ids]  # Shape: (sequence_length, embedding_dim)

# Print Aspects
print("Embedding Matrix:")
print(embedding_matrix)
print("\nInput IDs:")
print(input_ids)
print("\nOutput Embeddings:")
print(output)
```

---

##### Here is how you would use it directly from Alpaca:

```python
from Alpaca import Alpaca

alpaca = Alpaca()

VOCAB_SIZE = 4
EMBEDDING_DIM = 4

embedding_layer = alpaca.token_embedding(VOCAB_SIZE, EMBEDDING_DIM)

input_ids = torch.tensor([1, 2, 3, 0])

output = embedding_layer.forward(input_ids)

print(output)
```

---

#### Positional Encoding

The Positional Encoding Layer is accessed through the `Alpaca.pos_encoding()` method. This method takes in 2 params: `embedding_dim` the embedding dimension and `max_seq_len` the max sequence length. Positional encoding is used in Transformer models to provide information about the position of each token in a sequence. Since Transformers don’t have a built-in notion of word order (unlike RNNs), positional encodings are added to the token embeddings to give the model a sense of where each token is located relative to others.

---

##### Here is a code-based demonstration of the positional encoding concept using basic PyTorch:

```python
import torch
import math

# Predefined values
embedding_dim = 4  # Embedding dimension
max_seq_len = 10   # Maximum sequence length

# Initialize positional encodings matrix
position_encodings = torch.zeros(max_seq_len, embedding_dim)

# Fill the matrix with sine and cosine values
for pos in range(max_seq_len):
    for i in range(0, embedding_dim, 2):
        position_encodings[pos, i] = math.sin(pos / (10000 ** (i / embedding_dim)))
        if i + 1 < embedding_dim:
            position_encodings[pos, i + 1] = math.cos(pos / (10000 ** (i / embedding_dim)))

# Example input: Batch of 2 sequences, each of length 5
input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]])

# Retrieve positional encodings for the input sequence length
seq_len = input_ids.size(1)
output = position_encodings[:seq_len, :].unsqueeze(0).expand(input_ids.size(0), -1, -1)

print("Positional Encodings Matrix:")
print(position_encodings)
print("\nInput IDs:")
print(input_ids)
print("\nOutput Positional Encodings:")
print(output)
```

---

##### Here is how you would use it directly from Alpaca:

```python
from Alpaca import Alpaca

alpaca = Alpaca()

EMBEDDING_DIM = 4
MAX_SEQ_LEN = 10

pos_encoding_layer = alpaca.pos_encoding(EMBEDDING_DIM, MAX_SEQ_LEN)

input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]])

position_encodings = pos_encoding_layer.forward(input_ids)

print(position_encodings)
```

---

#### Encoder Block

The Encoder Block is made of:
1. A **Multi-Head Self-Attention Layer**, which is callable via the `Alpaca.multi_self_attention()` method. It takes in:
   - `d_model`: The model's dimensionality.
   - `num_heads`: The number of attention heads.
   - `masked`: A boolean flag to indicate whether masking is applied (used in the decoder).
2. A **Feed-Forward Network (FFN)**, which is callable via the `Alpaca.ffn()` method. It takes in:
   - `d_model`: The model's dimensionality.
   - `ff_dim`: The dimensionality of the hidden layer in the feed-forward network.
3. **Layer Normalization** and **Dropout** for stabilization and regularization.

---

##### Here is a code-based demonstration of the Encoder Block using basic PyTorch:

```python
import torch
import torch.nn as nn

# Predefined values
d_model = 4  # Model dimensionality
num_heads = 2  # Number of attention heads
ff_dim = 8  # Feed-forward hidden layer dimensionality
seq_len = 5  # Sequence length
batch_size = 2  # Batch size

# Input tensor (batch of 2 sequences, each of length 5)
x = torch.randn(batch_size, seq_len, d_model)

# Multi-Head Self-Attention
d_k = d_model // num_heads  # Dimension of each head

# Linear transformations for queries, keys, and values
W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)
W_v = nn.Linear(d_model, d_model, bias=False)
W_o = nn.Linear(d_model, d_model, bias=False)

# Compute queries, keys, and values
Q = W_q(x).view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
K = W_k(x).view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
V = W_v(x).view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)

# Scaled dot-product attention
scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)
attention = torch.softmax(scores, dim=-1)
attn_output = (attention @ V).transpose(1, 2).reshape(batch_size, seq_len, d_model)
attn_output = W_o(attn_output)

# Add & Norm (Layer Normalization and Dropout)
layer_norm1 = nn.LayerNorm(d_model)
dropout = nn.Dropout(0.1)
attn_output = layer_norm1(x + dropout(attn_output))

# Feed-Forward Network
linear1 = nn.Linear(d_model, ff_dim)
relu = nn.ReLU()
linear2 = nn.Linear(ff_dim, d_model)

ffn_output = linear1(attn_output)
ffn_output = relu(ffn_output)
ffn_output = linear2(ffn_output)

# Add & Norm (Layer Normalization and Dropout)
layer_norm2 = nn.LayerNorm(d_model)
output = layer_norm2(attn_output + dropout(ffn_output))

print("Input Tensor Shape:", x.shape)
print("Output Tensor Shape:", output.shape)
```

---

##### Here is how you would use it directly from Alpaca:

```python
from Alpaca import Alpaca

alpaca = Alpaca()

D_MODEL = 4
NUM_HEADS = 2
FF_DIM = 8

encoder_block = alpaca.encoder_block(D_MODEL, NUM_HEADS, FF_DIM)

input_tensor = torch.randn(2, 5, D_MODEL)  # Batch of 2 sequences, each of length 5
output = encoder_block.forward(input_tensor)

print(output)
```

---

### Decoder

---

#### Embedding

The Decoder's Embedding Layer works exactly the same as the Encoder's Embedding Layer. For details, refer to the [Embedding section](#Embedding).

---

#### Positional Encoding

The Decoder's Positional Encoding Layer works exactly the same as the Encoder's Positional Encoding Layer. For details, refer to the [Positional Encoding section](#Positional-Encoding).

---

#### Decoder Blocks

The Decoder Block is made of:
1. A **Multi-Head Self-Attention Layer**, which works the same as in the Encoder. For details, refer to the [Multi-Head Self-Attention section](#Multi-Head-Self-Attention).
2. A **Multi-Head Cross-Attention Layer**, which is unique to the Decoder.
3. A **Feed-Forward Network (FFN)**, which works the same as in the Encoder. For details, refer to the [Feed-Forward Network section](#Feed-Forward-Network).
4. **Layer Normalization** and **Dropout** for stabilization and regularization.

---

##### Multi-Head Cross-Attention

The **Multi-Head Cross-Attention Layer** is unique to the Decoder. It allows the Decoder to attend to the Encoder's output, enabling the model to incorporate information from the input sequence when generating the output sequence. It works similarly to Multi-Head Self-Attention but uses the Encoder's output for keys (`K`) and values (`V`), while the queries (`Q`) come from the Decoder's input.

---

##### Here is a code-based demonstration of Multi-Head Cross-Attention using basic PyTorch:

```python
import torch
import torch.nn as nn

# Predefined values
d_model = 4  # Model dimensionality
num_heads = 2  # Number of attention heads
seq_len = 5  # Sequence length
batch_size = 2  # Batch size

# Input tensors
x = torch.randn(batch_size, seq_len, d_model)  # Decoder input
encoder_output = torch.randn(batch_size, seq_len, d_model)  # Encoder output

# Dimension of each head
d_k = d_model // num_heads

# Linear transformations for queries, keys, and values
W_q = nn.Linear(d_model, d_model, bias=False)  # Query weights
W_k = nn.Linear(d_model, d_model, bias=False)  # Key weights
W_v = nn.Linear(d_model, d_model, bias=False)  # Value weights
W_o = nn.Linear(d_model, d_model, bias=False)  # Output weights

# Compute queries (from Decoder input)
Q = W_q(x).view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)

# Compute keys and values (from Encoder output)
K = W_k(encoder_output).view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
V = W_v(encoder_output).view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)

# Scaled dot-product attention
scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)
attention = torch.softmax(scores, dim=-1)

# Compute output
output = (attention @ V).transpose(1, 2).reshape(batch_size, seq_len, d_model)
output = W_o(output)

print("Decoder Input Shape:", x.shape)
print("Encoder Output Shape:", encoder_output.shape)
print("Cross-Attention Output Shape:", output.shape)
```

---

##### Here is how you would use it directly from Alpaca:

```python
from Alpaca import Alpaca

alpaca = Alpaca()

D_MODEL = 4
NUM_HEADS = 2

# Create the Multi-Head Cross-Attention layer
cross_attention_layer = alpaca.multi_cross_attention(D_MODEL, NUM_HEADS)

# Example inputs
decoder_input = torch.randn(2, 5, D_MODEL)  # Decoder input
encoder_output = torch.randn(2, 5, D_MODEL)  # Encoder output

# Forward pass
output = cross_attention_layer.forward(decoder_input, encoder_output)

print(output)
```

---

##### Feed-Forward Network

The Decoder's Feed-Forward Network works exactly the same as the Encoder's Feed-Forward Network. For details, refer to the [Feed-Forward Network section](#Feed-Forward-Network).

---

##### Layer Normalization and Dropout

The Decoder uses Layer Normalization and Dropout in the same way as the Encoder. For details, refer to the [Encoder Block section](#Encoder-Blocks).
```


```markdown
## Transformer Implementation

This section covers the practical implementation of the Alpaca Transformer, including how to create a Transformer model, use the Tokenizer, handle datasets, train the model, and perform inference.

---

### Creating an Alpaca Transformer

To create a Transformer model, use the `alpaca.new_transformer()` method. This method initializes and returns a Transformer with the specified parameters.

---

##### Here is how you create a Transformer:

```python
from Alpaca import Alpaca

# Initialize Alpaca
alpaca = Alpaca()

# Define parameters
VOCAB_SIZE = 10000  # Size of the vocabulary
D_MODEL = 512       # Dimensionality of the model
NUM_HEADS = 8       # Number of attention heads
FF_DIM = 2048       # Dimensionality of the feed-forward network
NUM_LAYERS = 6      # Number of encoder/decoder layers
MAX_SEQ_LEN = 128   # Maximum sequence length

# Create the Transformer
transformer = alpaca.new_transformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, FF_DIM, NUM_LAYERS, MAX_SEQ_LEN)

print(transformer)
```

---

### Tokenizer

The Tokenizer is a crucial component for converting text into tokens and vice versa. It supports creating vocabularies, tokenizing text, detokenizing tokens, and saving/loading vocabularies.

---

#### Accessing the Tokenizer

The Tokenizer is created automatically when you instantiate the `Alpaca` class. You can access it using:

```python
tokenizer = alpaca.tokenizer()
```

---

#### Tokenizer Methods

##### `tokenize(text, vocab=None, save_as_file=False, save_file_path='tokens.txt')`

- **Purpose**: Converts input text into tokens using the vocabulary.
- **Parameters**:
  - `text`: The input text to tokenize.
  - `vocab`: Optional. A pre-existing vocabulary to use. If not provided, the Tokenizer will create one.
  - `save_as_file`: If `True`, saves the tokens to a file.
  - `save_file_path`: The path to save the tokens file.
- **Returns**: A list of tokens.

##### Example:

```python
text = "Doing work is a lot of work!"
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)
```

---

##### `detokenize(tokenized, vocab=None, include_unknown=False)`

- **Purpose**: Converts tokens back into text.
- **Parameters**:
  - `tokenized`: A list of tokens to detokenize.
  - `vocab`: Optional. A pre-existing vocabulary to use. If not provided, the Tokenizer's current vocabulary is used.
  - `include_unknown`: If `True`, includes `<unk>` for unknown tokens.
- **Returns**: The detokenized text.

##### Example:

```python
detokenized_text = tokenizer.detokenize(tokens)
print("Detokenized Text:", detokenized_text)
```

---

##### `create_vocab(text, num_merges=5)`

- **Purpose**: Creates a vocabulary from the input text using Byte Pair Encoding (BPE).
- **Parameters**:
  - `text`: The input text to create the vocabulary from.
  - `num_merges`: The number of merge operations to perform.
- **Returns**: The created vocabulary.

##### Example:

```python
vocab = tokenizer.create_vocab(text)
print("Vocabulary:", vocab)
```

---

##### `load_vocab(vocab_path)`

- **Purpose**: Loads a vocabulary from a JSON file.
- **Parameters**:
  - `vocab_path`: The path to the vocabulary JSON file.
- **Returns**: The loaded vocabulary.

##### Example:

```python
vocab = tokenizer.load_vocab("vocab.json")
print("Loaded Vocabulary:", vocab)
```

---

##### `save_as_file(vocab_save_path='vocab.json', token_save_path='tokens.json')`

- **Purpose**: Saves the current vocabulary and tokens to JSON files.
- **Parameters**:
  - `vocab_save_path`: The path to save the vocabulary file.
  - `token_save_path`: The path to save the tokens file.

##### Example:

```python
tokenizer.save_as_file("my_vocab.json", "my_tokens.json")
```

---

##### Example Workflow:

```python
from Alpaca import Alpaca

# Initialize Alpaca and Tokenizer
alpaca = Alpaca()
tokenizer = alpaca.tokenizer()

# Example text
text = "Doing work is a lot of work!"

# Create vocabulary
vocab = tokenizer.create_vocab(text)

# Tokenize text
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

# Detokenize tokens
detokenized_text = tokenizer.detokenize(tokens)
print("Detokenized Text:", detokenized_text)

# Save vocabulary and tokens
tokenizer.save_as_file("vocab.json", "tokens.json")
```

---

### Creating an Alpaca Dataset

The `alpaca.dataset()` method creates a dataset from a text file. It tokenizes the text and prepares it for training.

---

##### Syntax:

```python
dataset = alpaca.dataset(txt_file, tokenizer=None, vocab=None, max_seq_len=512, merges=5000)
```

- **Parameters**:
  - `txt_file`: The path to the text file.
  - `tokenizer`: Optional. A Tokenizer object. If not provided, the default Tokenizer is used.
  - `vocab`: Optional. A pre-existing vocabulary. If not provided, the Tokenizer will create one.
  - `max_seq_len`: The maximum sequence length for the dataset.
  - `merges`: The number of merge operations for Byte Pair Encoding (BPE).
- **Returns**: A dataset object ready for training.

##### Example:

```python
# Create a dataset from a text file
dataset = alpaca.dataset("my_text_file.txt", max_seq_len=128)

print(dataset)
```

---

### Training an Alpaca Transformer

The `alpaca.train_model()` method trains the Transformer model. If no Transformer is provided, it uses the one stored in the Alpaca object.

---

##### Syntax:

```python
alpaca.train_model(epochs, train_dl, optimizer=torch.optim.Adam, transformer=None, loss_fn=nn.CrossEntropyLoss, lr=1e-4, validate_data=False, validation_data=None, wandb_tracking=False, lr_scheduler=False)
```

- **Parameters**:
  - `epochs`: The number of training epochs.
  - `train_dl`: The training DataLoader.
  - `optimizer`: The optimizer to use (default is `torch.optim.Adam`).
  - `transformer`: Optional. A Transformer model. If not provided, the default Transformer in the Alpaca object is used.
  - `loss_fn`: The loss function (default is `nn.CrossEntropyLoss`).
  - `lr`: The learning rate (default is `1e-4`).
  - `validate_data`: If `True`, performs validation during training.
  - `validation_data`: Optional. The validation DataLoader.
  - `wandb_tracking`: If `True`, enables Weights & Biases tracking.
  - `lr_scheduler`: If `True`, enables a learning rate scheduler.
- **Returns**: The trained Transformer model.

##### Example:

```python
# Train the model
alpaca.train_model(epochs=10, train_dl=train_dataloader, lr=1e-4, validate_data=True, validation_data=val_dataloader)
```

---

### Creating Predictions Using an Alpaca Transformer

The `alpaca.inference()` method generates predictions using the Transformer. If no state dictionary is provided, it uses the one stored in the Alpaca object.

---

##### Syntax:

```python
output = alpaca.inference(tokens, state_dict=None, detokenize=False, vocab=None)
```

- **Parameters**:
  - `tokens`: The input tokens for inference.
  - `state_dict`: Optional. A state dictionary for the model. If not provided, the default one in the Alpaca object is used.
  - `detokenize`: If `True`, returns the output as text. If `False`, returns tokens.
  - `vocab`: Optional. A vocabulary for detokenization. If not provided, the Tokenizer's vocabulary is used.
- **Returns**: The model's output (either tokens or text).

##### Example:

```python
# Perform inference
output = alpaca.inference(tokens, detokenize=True)
print("Model Output:", output)
```

---

### Summary of Workflow

```python
from Alpaca import Alpaca

# Initialize Alpaca
alpaca = Alpaca()

# Create a Transformer
transformer = alpaca.new_transformer(vocab_size=10000, d_model=512, num_heads=8, ff_dim=2048, num_layers=6, max_seq_len=128)

# Create a dataset
dataset = alpaca.dataset("my_text_file.txt", max_seq_len=128)

# Create a DataLoader
train_dataloader = DataLoader(dataset, batch_size=batch_size)

# Train the model
alpaca.train_model(epochs=10, train_dl=train_dataloader, lr=1e-4, validate_data=True, validation_data=val_dataloader)

# Perform inference
output = alpaca.inference(tokens, detokenize=True)
print("Model Output:", output)
```
```