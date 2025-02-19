# Alpaca Transformer Model from Scratch

This project demonstrates the process of building a transformer model from scratch, utilizing PyTorch for deep learning. It covers the essential components of transformer architectures, such as tokenization, embedding layers, multi-head self-attention, and the training pipeline. This model is designed for educational purposes to help users understand and implement transformers without relying on pre-built models.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
3. [Usage](#usage)
   - [Data Preparation](#data-preparation)
   - [Training](#training)
4. [Model Architecture](#model-architecture)
5. [Contributing](#contributing)
6. [License](#license)

## Project Overview
The **Alpaca Transformer** is a custom-built transformer model designed from scratch to perform token classification tasks. It includes a custom tokenizer, vocabulary creation, tokenization process, and the full transformer architecture. The model is implemented in PyTorch, using standard transformer building blocks such as embedding layers, multi-head self-attention, and position encodings.

Key Features:
- Tokenizer and vocabulary creation from scratch.
- Transformer architecture with multi-head self-attention.
- Training pipeline to fine-tune the model.
- Modular and extensible codebase for educational purposes.
- All the tools have been made into easy to use methods in the 'alpaca.py' file

## Getting Started

### Prerequisites
Before you begin, ensure that you have the following installed on your system:
- **Python** (>= 3.7)
- **PyTorch** (>= 1.7.0)
- **CUDA** (for GPU acceleration, optional but recommended)

I personally used Python=3.12.7 and PyTorch=2.6.0-Cuda18 so if you're having issues try it.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RazielMoesch/alpaca.git
   cd alpaca

   or 
   
   pip install alpaca-transformer
   


## Usage

### Data Preperation
To train the model, you’ll need a dataset in text format. Each line in the dataset represents a sentence to be tokenized. The tokenizer will process the text into tokens, which are then padded to a uniform length (e.g., 512 tokens).
Prepare your dataset in a text file (eg., data.txt) where each line represents a sentence
The tokenization happens automatically when you use 'alpaca.dataset'
At the bottom of majority of the files there are left over test examples feel free to use them to understand how each file works.

### Training
To train the transformer model, you can follow these step:
1. Define your models optimizer, loss_fn and epochs
2. use 'alpaca.train()' this takes in multiple params.

## Model Architecture
This model follows a standard transformer architecture as its backbone:
- Tokenizer - Transforms text into interpratble tokens
- Embedding Layer - Maps tokens to vectors 
- Multi-Head Self Attention - Allows the model to focus on different parts of the input
- Feed-Forward-Network - A Linear,ReLU,Linear layer
- Positional-Encoding - Use sin and cos funcs to give the model info about the order of the sequence
- Stacking - Stack Many of these in Encoder and Decoder Layers to achieve a Transformer

## License
This is under a creative commons license just look at the file if you want specifics
Please don't outright steal. Only restriction.
