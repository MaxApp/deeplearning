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
    

if __name__ == "__main__":

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Create a simple encoder block with small dimensions for demonstration
    # encoder_demo = EncoderBlock(embedding_dim=4, nhead=1, ffn_mult=4)


