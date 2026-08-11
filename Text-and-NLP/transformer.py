import math
import os
import re
import urllib.request
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class SimpleTokenizer:
    """splits by words and lowercases"""
    def __init__(self):
        pass

    def __call__(self, text):
        return re.findall(r'\b\w+\b', text.lower())


class MyDataset(Dataset):

    def __init__(self, sentences, word2idx):

        SEQ_LEN = 3   # length of input sequence for each example

        # convert to token id
        encoded_sentences = []
        for s in sentences:
            tokens = tokenizer(s)
            ids = [word2idx.get(tok, word2idx['<unk>']) for tok in tokens]
            encoded_sentences.append(ids)

        # sliding window dataset (inputs, targets)
        inputs = []
        targets = []
        for ids in encoded_sentences:
            for i in range(len(ids) - SEQ_LEN):
                window = ids[i:i+SEQ_LEN]    
                target = ids[i+SEQ_LEN]
                inputs.append(window)
                targets.append(target)

        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class MyAttention(nn.Module):
    
    def __init__(self, d):
        super().__init__()
        self.to_q = nn.Linear(d, d, bias=False)
        self.to_k = nn.Linear(d, d, bias=False)
        self.to_v = nn.Linear(d, d, bias=False)

    def forward(self, x):
        Q = self.to_q(x)
        K = self.to_k(x)
        V = self.to_v(x)

        # scaled dot-product: (Q @ K^T) / sqrt(dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))

        # apply softmax for each row
        attn = F.softmax(scores, dim=-1)

        # combine with V
        output = torch.matmul(attn, V)
        return output, attn

def build_vocab(sentences, tokenizer, min_freq=1):
    # calculate in global range
    counter = Counter()
    for s in sentences:
        counter.update(tokenizer(s))

    # add <pad>,<unk>
    vocab = ['<pad>', '<unk>'] + [w for w, c in counter.items() if c >= min_freq]

    # create mappings
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}

    return vocab, word2idx, idx2word




if __name__ == "__main__":

    # sample sentences
    sentences = [
        "I drove car to the park with my mother",
        "My mother sits in the car",
        "My mother drove the car",
        "the car is red",
        "I go to the park",
        "My mother go to the park",
        "my mother and I go to the restaurant",
        "my brother can drove the car",
        "I can drove horse",
        "my brother drove hosre with me",
        "my mother bought a horse for me"
    ]

    tokenizer = SimpleTokenizer()
    vocab, word2idx, idx2word = build_vocab(sentences, tokenizer)

    dataset = MyDataset(sentences, word2idx)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # for inputs, targets in loader:
    #     for inp, tgt in zip(inputs, targets):
    #         inp_words = [idx2word[i.item()] for i in inp]
    #         tgt_word = idx2word[tgt.item()]
    #         print(f"Input: {inp_words}  -> Target: {tgt_word}")

    