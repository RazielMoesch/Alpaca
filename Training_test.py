from .alpaca import Alpaca

alpaca = Alpaca()

text = 'Hello'

tokenizer = alpaca.tokenizer
tokenizer.create_vocab(text)

alpaca.tokenizer.save_as_file()