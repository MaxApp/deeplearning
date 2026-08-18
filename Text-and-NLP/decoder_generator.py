import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math


class DecoderBlock(nn.Module):

    """
    customized decoder-ONLY with no cross-attention
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
        attn_out, attn_out_weights = self.mha(x_norm, x_norm, x_norm, attn_mask=causal_mask)
        x = x + self.dropout1(attn_out)
        
        # FFN
        ffn_in = self.layer_norm2(x)
        ffn_out = self.ffn(ffn_in)
        x = x + self.dropout2(ffn_out)
        
        return x


class Generator(nn.Module):
    """
    Decoder built with TransformerDecoder layers
    """
    def __init__(self, vocab_size, embedding_dim=128, nhead=6, num_layers=2,
                 dim_feedforward=512, max_len=512, dropout=0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.token_emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(max_len, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # TransformerDecoderLayer has both self-attention and cross-attention
        dec_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers)
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.layer_final = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, src):

        padding_mask = (src == 0)
        causal_mask = create_causal_mask(src.size(1))
        
        src = self.token_emb(src)
        src = src + self.pos_enc(src)
        src = self.dropout(src)
        
        # decoder for generation
        src = self.transformer_decoder(
            tgt=src,  # target
            memory=src, # source inputs
            tgt_mask=causal_mask,
            memory_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask
        )
        
        output = self.layer_final(src)
        return output

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

def create_causal_mask(size: int, is_bool=True):
    if is_bool:
        # fill with type of bool
        mask = torch.ones(size, size)
        mask = torch.triu(mask, diagonal=1).bool()
    else:
        # fill with type of float
        mask = torch.full((size, size), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
    return mask

if __name__ == "__main__":

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # test
    vocab_size = 100  # Small vocabulary for demo
    d_model = 128
    nhead = 4
    num_layers = 2

    decoder = Generator(
        vocab_size=vocab_size,
        embedding_dim=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=512,
        dropout=0.1
    )
    decoder.eval()

    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))  # random token ids

    print(f"Input shape: {input_ids.shape}")
    print(f"Sample input: {input_ids[0]}")

    with torch.no_grad():
        output = decoder(input_ids)

    print(f"Output shape: {output.shape}")
    print(f"batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}")