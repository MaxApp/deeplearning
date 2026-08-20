import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
from pathlib import Path
from collections import Counter
import random
import os
import re

from utils import IMDBTokenizer


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

class IMDBReviewDataset(Dataset):
    def __init__(self, data_dir, tokenizer, train=True):
        data_path = os.path.join(data_dir, "train" if train else "test")
        self.reviews = []

        # load positive
        data_dir = os.path.join(data_path, "pos")
        data_files = os.listdir(data_dir)
        for filename in data_files:
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                self.reviews.append(tokenizer(content))

        # load negative
        data_dir = os.path.join(data_path, "neg")
        data_files = os.listdir(data_dir)
        for filename in data_files:
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                self.reviews.append(tokenizer(content))

    def __len__(self):
            return len(self.reviews)
    
    def __getitem__(self, idx):
        return self.reviews[idx], self.reviews[idx]


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

def train_model(model, train_dataloader, optimizer, loss_func, vocab_size, num_epoch):
     for epoch in range(num_epoch):
        model.train()
        epoch_losses = []  # Track losses for averaging
        for reviews, labels in train_dataloader:
            optimizer.zero_grad()
            logits = model(reviews)

            # shift for predict
            shifted_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
            shifted_labels = labels[:, 1:].contiguous().view(-1)
            loss = loss_fn(shifted_logits, shifted_labels)
            # Backward pass
            loss.backward()
            
            # Gradient clipping (ADD THIS!)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            # Track loss
            epoch_losses.append(loss.item())
    
        # Calculate average loss - simple mean
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1:2d}: avg loss = {avg_loss:.4f}")

def generate_tokens(model, prompt_ids, max_length=100, temperature=1.0, 
                   top_k=50, top_p=0.95, repetition_penalty=1.2, 
                   eos_token_id=None, device='cpu'):
    """
    Advanced token generation with multiple sampling strategies.
    
    Args:
        model: The trained model
        prompt_ids: Starting token IDs (list or tensor)
        max_length: Maximum length to generate
        temperature: Controls randomness (0.1=conservative, 2.0=creative)
        top_k: Keep only top k tokens (0=disabled)
        top_p: Nucleus sampling threshold (0.95=default)
        repetition_penalty: Penalty for repeated tokens
        eos_token_id: End of sequence token ID
        device: Device to run on
    
    Returns:
        Generated token IDs as tensor
    """
    model.eval()
    
    # Handle different input formats
    if isinstance(prompt_ids, list):
        prompt_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    elif len(prompt_ids.shape) == 1:
        prompt_ids = prompt_ids.unsqueeze(0).to(device)
    else:
        prompt_ids = prompt_ids.to(device)
    
    generated = prompt_ids.clone()
    past_tokens = list(prompt_ids[0].cpu().numpy())

    # print(f"prompt_ids: {prompt_ids}")
    # print(f"generated: {generated}")
    # print(f"past_tokens: {past_tokens}")
    for _ in range(max_length - len(prompt_ids[0])):
        logits = model(generated)
        
        # Get the last token's logits
        next_token_logits = logits[0, -1, :].float()
        
        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            # Penalize all previously generated tokens
            for token_id in set(past_tokens):
                next_token_logits[token_id] /= repetition_penalty
            
            # Extra penalty for very recent tokens
            if len(past_tokens) > 3:
                for token_id in past_tokens[-3:]:
                    next_token_logits[token_id] /= 1.5
        
        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, min(top_k, len(next_token_logits)))[0][-1]
            next_token_logits[indices_to_remove] = -float('inf')
        
        # Apply nucleus (top-p) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float('inf')
        
        # Sample from the distribution
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        # print(f"next_token : {next_token}")
        
        # Append to generated sequence
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        past_tokens.append(next_token.item())
        yield next_token.unsqueeze(0)
        
        # Stop if we hit the EOS token
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break


def generate_text(model, prompt, tokenizer,
                 max_length=100, temperature=0.8, top_k=50, top_p=0.95,
                 repetition_penalty=1.2, device='cpu'):
    # tokenize prompt
    if not prompt or prompt.isspace():
        # start with 'the' if no prompt
        prompt = 'the'
    # convert to indices
    prompt_ids = tokenizer.encode(prompt, with_padding=False)
    
    # ensure have at least one token
    if not prompt_ids:
        prompt_ids = [2] # '<sos>'
    
    # eos token IDs
    eos_token_id = 3 # '<eos>'
    
    for next_id in generate_tokens(
        model,
        prompt_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_token_id,
        device=device
    ):
        yield tokenizer.decode(next_id, is_tensor=True)[0]
    
    # Fix punctuation spacing
    # text = text.replace(' ,', ',').replace(' .', '.').replace(' !', '!')
    # text = text.replace(' ?', '?').replace(' ;', ';').replace(' :', ':')
    # text = text.replace(' \'', '\'').replace('\' ', '\'')
    # text = text.replace(' \n ', '\n').replace('\n ', '\n')
    
    # return text.strip()

class CustomTokenizer:
    """handles words and punctuation"""
    def __init__(self, token_len, max_vocab_size=10000):
            # the max length of tokenized inputs
            self.token_len = token_len
            # the max vocabulary size, not exceed
            self.max_vocab_size = max_vocab_size
            # special tokens were reserved
            self.word_to_idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
            self.idx_to_word = {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}
            self.word_freq = Counter()

    def __call__(self, text):
        """encode to tensors for model"""
        return torch.tensor(self.encode(text), dtype=torch.long)
        
    def tokenize(self, text):
        """splits inputs for human readable"""
        text = text.lower()
        # remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # remove line breaks
        text = text.replace('\n', '')
        # tokenize words and punctuation
        return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)

    def encode(self, text, with_padding=True):
        """encode tokenized words to indicies, including <pad>,<unk>,<sos>,<eos>"""

        words = self.tokenize(text)[:self.token_len-2]  # reserve space for SOS/EOS tokens
        
        # start with '<sos>': 2
        indices = [2]
        
        for word in words:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                # '<unk>': 1
                indices.append(1)

        if not with_padding:
            return indices
        
        # '<eos>': 3
        indices.append(3)
        
        # '<pad>': 0
        while len(indices) < self.token_len:
            indices.append(0)
        
        return indices[:self.token_len]

    def decode(self, sequence, is_tensor=False):
        """convert inidicies back to tokens for readable"""
        if is_tensor:
            all_tokens = [self.idx_to_word[i.item()] for i in sequence]
        else:
            all_tokens = [self.idx_to_word[i] for i in sequence]
        return all_tokens
        
    def build_vocabulary(self, directory:str, min_freq=2):
        # read all the .txt files in the path
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Path Not Exist: {directory}")

        txt_files = list(dir_path.rglob('*.txt'))
        for f in txt_files:
            with f.open('r', encoding='utf-8') as f:
                review = f.read()
                words = self.tokenize(review)
                # count word frequencies
                self.word_freq.update(words)
        
        # most common words within (vocab_size - 4),
        # reserve 4 spots for special tokens
        most_common = self.word_freq.most_common(self.max_vocab_size - 4)

        # build w2i, i2w
        idx = 4  # after special tokens
        # uncollected_words = []
        for word, freq in most_common:
            if freq >= min_freq:
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                idx += 1
        
        print(f"Vocabulary size: {len(self.word_to_idx)}")

    def size(self):
        return len(self.word_to_idx)

if __name__ == "__main__":

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # test
    # vocab_size = 100  # Small vocabulary for demo
    d_model = 128
    nhead = 4
    num_layers = 2

    token_len = 512 # hyper parameters need to changed
    corpus_dir = "E:/PDF/pytorch/C3M3/imdb" # change to your own

    tokenizer = CustomTokenizer(token_len=token_len)
    tokenizer.build_vocabulary(directory=corpus_dir, min_freq=1)

    train_dataset = IMDBReviewDataset(data_dir="E:/PDF/pytorch/C3M3/imdb", tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)

    # Setup training components
    model = Generator(
        vocab_size=tokenizer.size(),
        embedding_dim=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=512,
        dropout=0.1
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.word_to_idx['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_model(model=model, train_dataloader=train_dataloader, vocab_size=tokenizer.size(),
                optimizer=optimizer, loss_func=loss_fn, num_epoch=5)

    # test prompt 
    import time
    prompt = "The Film"
    print(f"{prompt}", end="", flush=True)
    for next_token in generate_text(model=model,
                                    prompt=prompt,
                                    tokenizer=tokenizer,
                                    max_length=100,
                                    temperature=0.8,
                                    ):
        if next_token not in ['<pad>', '<unk>', '<sos>', '<eos>']:
            print(f" {str(next_token).strip()}", end="", flush=True)
            time.sleep(1)
            

