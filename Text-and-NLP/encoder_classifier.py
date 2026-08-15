import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import re
import random


class EncoderBlock(nn.Module):

    def __init__(self, embedding_dim=8, nhead=1, ffn_mult=4):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        # multi-head attention
        self.mha = nn.MultiheadAttention(embedding_dim, nhead, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)        
        # feedforward layer
        hidden = ffn_mult * embedding_dim
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embedding_dim)
        )
    
    def forward(self, x):        
        x_norm = self.layer_norm1(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        # residual
        x = x + attn_out
        
        ffn_in = self.layer_norm2(x)
        ffn_out = self.ffn(ffn_in)
        # residual
        x = x + ffn_out
        
        return x


class PositionalEncoding(nn.Module):
    """
    This module could be saved with transformer model,
    it use pre-created sin/cos encodings instead of creating them on-the-fly
    """
    
    def __init__(self, max_len, d_model):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        
        # pre-create maximum positional encoding matrix
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # sinusoidal pattern, div_term is 1/2 dimension of d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(torch.log(torch.tensor(10000.0)) / d_model))
        
        # apply sin to even indices, 1/2 dimensions
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        # apply cos to odd indices, 1/2 dimensions
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        
        # register as buffer (not trained, but saved with model)
        self.register_buffer('pos_enc', pos_enc.unsqueeze(0))
        
    def forward(self, x):
        """
        truncate pre-created positional encodings for input tensor.
        input seq_len should <= max_len
        returns same shape as input
        """
        seq_len = x.size(1) # actual input length
        return self.pos_enc[:, :seq_len, :]


class IMDBVocabManager:

    def __init__(self, vocab_size=10000):
        # the target vocabulary size, not exceed
        self.vocab_size = vocab_size
        # special tokens
        self.word_to_idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.idx_to_word = {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}
        self.word_freq = Counter()
    
    def build_vocab(self, texts, min_freq=2):
        """build vocabulary"""
        # count word frequencies
        for text in texts:
            words = self.tokenize(text)
            self.word_freq.update(words)
        
        # most common words within (vocab_size - 4),
        # reserve 4 spots for special tokens
        most_common = self.word_freq.most_common(self.vocab_size - 4)
        
        idx = 4  # after special tokens
        for word, freq in most_common:
            if freq >= min_freq:
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                idx += 1
        
        # print(f"Vocabulary size: {len(self.word_to_idx)}")
    
    def tokenize(self, text):
        text = text.lower()
        # remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # remove all punctuations, digits, keep only letters and spaces
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        return words
        
    def encode(self, text, max_len=256):

        words = self.tokenize(text)[:max_len-2]  # Leave space for SOS/EOS tokens
        
        # start with '<sos>': 2
        indices = [2]
        
        for word in words:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                # '<unk>': 1
                indices.append(1)
        
        # '<eos>': 3
        indices.append(3)
        
        # '<pad>': 0
        while len(indices) < max_len:
            indices.append(0)
        
        return indices[:max_len]

    

if __name__ == "__main__":

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Create a simple encoder block with small dimensions for demonstration
    # encoder_demo = EncoderBlock(embedding_dim=4, nhead=1, ffn_mult=4)


