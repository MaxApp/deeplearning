import torch
from transformers import BertTokenizerFast, AutoTokenizer

sentences = [
    'I love my dog a lot',
    'I don\'t love other\'s cat'
]

# Define the local directory where the tokenizer is saved
local_tokenizer_path = "./bert_tokenizer_local"

# Initialize the tokenizer from the local directory
tokenizer = BertTokenizerFast.from_pretrained(local_tokenizer_path)

# Tokenize the sentences and encode them
encoded_inputs = tokenizer(sentences, padding=True, 
                           truncation=True, return_tensors='pt')

# To see the tokens for each input (helpful for understanding the output)
tokens = [tokenizer.convert_ids_to_tokens(ids)
          for ids in encoded_inputs["input_ids"]]

# Get the model's vocabulary (mapping from tokens to IDs)
word_index = tokenizer.get_vocab() # For BertTokenizerFast, get_vocab() returns the vocab

# Print the human-readable `tokens` for each sentence
print("Tokens:", tokens)

print("\nToken IDs:", encoded_inputs['input_ids'])