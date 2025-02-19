from .alpaca import Alpaca

alpaca = Alpaca()

txt_file = 'example.txt'

tokenizer = alpaca.tokenizer

test_dataset = alpaca.dataset(txt_file, tokenizer=tokenizer)

print(test_dataset)