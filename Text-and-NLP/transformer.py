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

import utils


class SimpleTokenizer:
    """splits by words and lowercases"""
    def __init__(self):
        pass

    def __call__(self, text):
        return re.findall(r'\b\w+\b', text.lower())


class MyDataset(Dataset):
    """
    Build training data for predict next word.
    generate with `SEQ_LEN` words(input) paired with next word(target)
    """
    def __init__(self, sentences, word2idx):
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
            for i in range(len(ids) - seq_len):
                window = ids[i:i+seq_len]    
                target = ids[i+seq_len]
                inputs.append(window)
                targets.append(target)

        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class MyAttentionModel(nn.Module):
    
    def __init__(self, vocab_size, token_dim, max_seq_len, represent_dim):
        super().__init__()
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, token_dim)
        # position embedding layer
        self.pos_embedding = nn.Embedding(max_seq_len, token_dim)

        # Q,K,V
        self.to_q = nn.Linear(token_dim, represent_dim, bias=False)
        self.to_k = nn.Linear(token_dim, represent_dim, bias=False)
        self.to_v = nn.Linear(token_dim, represent_dim, bias=False)

        # FC layer
        self.fc = nn.Linear(represent_dim, vocab_size)

    def forward(self, token_ids):
        # because we have an uniform dimensions for inputs
        batch_size, seq_len = token_ids.shape
        # position indices for each input sequence by order in each batch
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        tk_emb = self.embedding(token_ids)
        pos_emb = self.embedding(positions)
        input_vecs = tk_emb + pos_emb   # sum word and position embeddings
        
        Q = self.to_q(input_vecs)
        K = self.to_k(input_vecs)
        V = self.to_v(input_vecs)

        # scaled dot-product: (Q @ K^T) / sqrt(dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))

        # apply softmax for each row
        attn_weights = F.softmax(scores, dim=-1)

        # combined with V
        attn_out = torch.matmul(attn_weights, V)

        last_hidden = attn_out[:, -1, :] # get the last word weighted values for prediction
        logits = self.fc(last_hidden)

        return logits, attn_weights

class Vocabulary:

    def __init__(self, tokenizer, min_freq=1) -> None:
        self.tokenizer = tokenizer
        self.min_freq = min_freq
        # init with <pad>,<unk>
        self.vocab = ["<pad>","<unk>"]
        self.word2idx = {}
        self.idx2word = {}

    def __len__(self):
        return len(self.vocab)

    def build_vocab(self, sentences):
        # calculate in global range
        counter = Counter()
        for s in sentences:
            counter.update(self.tokenizer(s))

        self.vocab += [w for w, c in counter.items() if c >= self.min_freq]

        # create mappings
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}

        return self.vocab, self.word2idx, self.idx2word

    # def get_token_ids(self, sentences):
    #     tk_sent = []
    #     for s in sentences:
    #         tk = [word2idx.get(tk, word2idx.get('<unk>')) for tk in self.tokenizer(s)]
    #         tk_sent.append(tk)
    #     return tk_sent
            

def train_model(model, loader, loss_fn, optimizer, epochs=20, device='cpu'):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0) # accumulate total loss
        avg_loss = total_loss / len(loader.dataset)    # average loss
        print(f"Epoch {epoch+1}: avg loss = {avg_loss:.4f}")


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
    seq_len = 3   # length of input sequence for training example

    tokenizer = SimpleTokenizer()
    vocabulary = Vocabulary(tokenizer)
    vocab, word2idx, idx2word = vocabulary.build_vocab(sentences)

    # token_ids = vocabulary.get_token_ids(sentences)
    # print(f"{token_ids}")

    dataset = MyDataset(sentences, word2idx)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # embedding the word
    torch.manual_seed(42)
    embedding_dim = 4
    qkv_dim = 6
    max_length = max(len(s) for s in sentences)

    attention_model = MyAttentionModel(vocab_size=len(vocabulary), 
                                       token_dim=embedding_dim, 
                                       max_seq_len=max_length, 
                                       represent_dim=qkv_dim)

    optimizer = optim.AdamW(attention_model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train model
    train_model(attention_model, loader, loss_fn, optimizer, epochs=100, device=device)

    print(f"== Training Finished ==")
    # plot the heat map to show attentions between words after training
    test_sentence = "I and mother"
    sample_tokens = tokenizer(test_sentence)
    max_tokens = 5 # how many new words to generate
    attention_model.eval()
    for _ in range(max_tokens):
        window = sample_tokens[-seq_len:] if len(sample_tokens) >= seq_len \
                     else ['<pad>'] * (seq_len - len(sample_tokens)) + sample_tokens
        # print(f"generate with window: '{window}'")
        input_ids = torch.tensor([[word2idx.get(w, word2idx['<unk>']) for w in window]], dtype=torch.long).to(device)

        with torch.no_grad():
            logits, attn = attention_model(input_ids)
            # print(f"out: {logits}")
            # print(f"weights: {attn}")
            next_id = logits.argmax(dim=-1).item()

        next_word = idx2word[next_id]
        sample_tokens.append(next_word) 
        print(f"{sample_tokens}")

    # print(f"attention weights:\n {attn[0].detach().numpy()}") # [5,5]
    # print(f"weighted V:\n {out[0].detach().numpy()}") # [5,6]

    # utils.plot_attention(attn, tokens)

    # for inputs, targets in loader:
    #     for inp, tgt in zip(inputs, targets):
    #         inp_words = [idx2word[i.item()] for i in inp]
    #         tgt_word = idx2word[tgt.item()]
    #         print(f"Input: {inp_words}  -> Target: {tgt_word}")

    