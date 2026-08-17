import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math


class DecoderBlock(nn.Module):

    """
    customized decoder with no cross-attention
    """

    def __init__(self, embedding_dim: int, nhead: int = 1, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        
        # layer normalization
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        # multi-head attention
        self.mha = nn.MultiheadAttention(embedding_dim, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        # layer normalization
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embedding_dim)
        )
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, causal_mask):
        x_norm = self.layer_norm1(x)
        # masked self-attention with residual
        attn_out, _ = self.mha(x_norm, x_norm, x_norm, attn_mask=causal_mask)
        x = x + self.dropout1(attn_out)
        
        # FFN
        ffn_in = self.layer_norm2(x)
        ffn_out = self.ffn(ffn_in)
        x = x + self.dropout2(ffn_out)
        
        return x





if __name__ == "__main__":

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)