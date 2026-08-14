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
    
    def __init__(self, max_len, d_model):
        """
        Initialize positional encoding matrix.
        
        Args:
            max_len (int): Maximum sequence length the model will handle
                          (e.g., 100 for sentences up to 100 tokens)
            d_model (int): Dimension of the model's embeddings 
                          (e.g., 256 or 512 - must match embedding size)
        
        Creates a fixed sinusoidal pattern matrix of shape [max_len, d_model]
        where each row represents the positional encoding for that position.
        """
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Create div_term for the sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(torch.log(torch.tensor(10000.0)) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices  
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not trained, but saved with model)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Return positional encodings for the input sequence length.
        
        Args:
            x (Tensor): Token embeddings of shape [batch_size, seq_len, d_model]
                       where seq_len <= max_len from initialization
        
        Returns:
            Tensor: Positional encodings of shape [batch_size, seq_len, d_model]
                   (same shape as input, ready to be added to embeddings)
        
        Example:
            If x represents embeddings for "I love cats" (3 tokens):
            - Input x shape: [batch_size, 3, 256]
            - Output shape: [batch_size, 3, 256]
            - Returns positions 0, 1, 2 encoded as 256-dim vectors
        """
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]
    

if __name__ == "__main__":

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Create a simple encoder block with small dimensions for demonstration
    encoder_demo = EncoderBlock(embedding_dim=4, nhead=1, ffn_mult=4)

    # Create a sample input: (batch_size=2, sequence_length=3, d_model=4)
    sample_input = torch.randn(2, 3, 4)

    print("Input shape:", sample_input.shape)
    print("Input tensor:\n", sample_input)

    # Pass through encoder block
    output = encoder_demo(sample_input)

    print("\nOutput shape:", output.shape)
    print("Output tensor:\n", output)

    # Notice that the shape remains the same
    print("\nShape preserved: Input shape == Output shape:", sample_input.shape == output.shape)