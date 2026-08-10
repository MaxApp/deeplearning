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
    """Splits on whitespace and lowercases, with optional regex for real word tokens."""
    def __init__(self):
        pass

    def __call__(self, text):
        # Option 1: Basic split (uncomment to just split on spaces)
        # return text.lower().split()
        # Option 2: More robust - returns only word tokens (ignores punctuation)
        return re.findall(r'\b\w+\b', text.lower())

def build_vocab(sentences, tokenizer, min_freq=1):
    counter = Counter()  # Counter to count word frequencies in all sentences
    for sent in sentences:
        counter.update(tokenizer(sent))  # Tokenize sentence and add token counts

    # Start vocab with special tokens, then add words meeting min_freq threshold
    vocab = ['<pad>', '<unk>'] + [w for w, c in counter.items() if c >= min_freq]

    # Create a mapping from word to unique index
    word2idx = {w: i for i, w in enumerate(vocab)}

    # Create a mapping from index back to word (inverse of word2idx)
    idx2word = {i: w for i, w in enumerate(vocab)}

    # Return the vocab list and the two dictionaries
    return vocab, word2idx, idx2word

# Using our sample sentences and tokenizer
sentences = [
    "the dog chased the cat",
    "the cat chased the mouse",
    "the dog ran fast",
    "the mouse ran fast",
    "the cat lay down"
]

tokenizer = SimpleTokenizer()                 # Define the tokenizer (splits into lowercase words)
vocab, word2idx, idx2word = build_vocab(sentences, tokenizer)  # Build vocab & mappings