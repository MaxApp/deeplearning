import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import re
import random

class EncoderBlock(nn.Module):
    def __init__(self, d_model=4, nhead=1, ffn_mult=4):
        super().__init__()
        # Layer normalization before attention
        self.ln1 = nn.LayerNorm(d_model)
        # Multi-head self-attention
        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        # Layer normalization before feed-forward
        self.ln2 = nn.LayerNorm(d_model)        
        # Feed-forward network with expansion
        hidden = ffn_mult * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_model)
        )
    
    def forward(self, x):        
        # First sub-layer: Multi-head attention with residual connection
        x_norm = self.ln1(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        x = x + attn_out  # Residual connection
        
        # Second sub-layer: Feed-forward with residual connection
        ffn_in = self.ln2(x)
        ffn_out = self.ffn(ffn_in)
        x = x + ffn_out  # Residual connection
        
        return x