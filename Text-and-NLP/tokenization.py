from transformers import BertTokenizerFast,GPT2TokenizerFast, AutoTokenizer

def get_tokenizer(tk_name):

    # `AutoTokenizer` is a better way in product, 
    # but for practice we specified tokenizer's type respectively
    if tk_name and tk_name.strip().lower() == "gpt":
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        # should set `eos_token` as padding for GPT2
        tokenizer.pad_token = tokenizer.eos_token
    elif tk_name and tk_name.strip().lower() == "bert":
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tk_name)

    return tokenizer


if __name__ == "__main__":

    sentences = [
        'I\'m feeling happy today because doing deeplearning',
        'Don\'t drop garbage anywhere in dinning room~'
    ]

    tokenizer = get_tokenizer("bert")
    # tokenize the sentences and encode
    encoded_inputs = tokenizer(sentences, padding=True, 
                               truncation=True, return_tensors='pt')
    # convert encoded ids to tokens
    tokens = [tokenizer.convert_ids_to_tokens(ids)
              for ids in encoded_inputs["input_ids"]]
    print("BERT Tokens:", tokens)
    print("BERT Token IDs:", encoded_inputs['input_ids'])

    tokenizer = get_tokenizer("gpt")
    # tokenize the sentences and encode
    encoded_inputs = tokenizer(sentences, padding=True, 
                               truncation=True, return_tensors='pt')
    # convert encoded ids to tokens
    tokens = [tokenizer.convert_ids_to_tokens(ids)
              for ids in encoded_inputs["input_ids"]]
    print("GPT Tokens:", tokens)
    print("GPT Token IDs:", encoded_inputs['input_ids'])

    