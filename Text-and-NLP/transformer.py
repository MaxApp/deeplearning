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
    
    def __init__(self, vocab_size, embedding_dim, max_seq_len, represent_dim):
        super().__init__()
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # position embedding layer
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)

        # Q,K,V
        self.to_q = nn.Linear(embedding_dim, represent_dim, bias=False)
        self.to_k = nn.Linear(embedding_dim, represent_dim, bias=False)
        self.to_v = nn.Linear(embedding_dim, represent_dim, bias=False)

        # FC layer
        self.fc = nn.Linear(represent_dim, vocab_size)

    def forward(self, token_ids):
        # simplified because we have an uniform dimensions for inputs
        batch_size, seq_len = token_ids.shape
        # position indices for each input sequence by order in each batch
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        attn_weights, attn_out = self.cal_attn(token_ids, positions)
        # get the last word weighted values for prediction
        last_hidden = attn_out[:, -1, :]
        logits = self.fc(last_hidden)

        return logits, attn_weights

    def cal_attn(self, token_ids, positions):
        # create embeddings
        tk_emb = self.embedding(token_ids)
        pos_emb = self.pos_embedding(positions)
        padding_mask = (token_ids == 0).unsqueeze(1)
        input_vecs = tk_emb + pos_emb   # sum word and position embeddings
        
        Q = self.to_q(input_vecs)
        K = self.to_k(input_vecs)
        V = self.to_v(input_vecs)

        # scaled dot-product: (Q @ K^T) / sqrt(dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))

        scores = scores.masked_fill(padding_mask, float('-inf'))
        # apply softmax for each row
        attn_weights = F.softmax(scores, dim=-1)

        # dot-product with V
        attn_out = torch.matmul(attn_weights, V)
        return attn_weights,attn_out

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

    def sentence_to_idx(self, sentence):
        tokens = self.tokenizer(sentence)
        ids = [word2idx.get(tok, word2idx['<unk>']) for tok in tokens]
        return ids, tokens

    # def get_token_ids(self, sentences):
    #     tk_sent = []
    #     for s in sentences:
    #         tk = [word2idx.get(tk, word2idx.get('<unk>')) for tk in self.tokenizer(s)]
    #         tk_sent.append(tk)
    #     return tk_sent
            

def train_model(model, loader, loss_fn, optimizer, epochs=20, device='cpu'):
    print(f"--- Training Started ---")
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
    print(f"--- Training Finished ---")


def predict_next_n_words(model, given_sentence, max_predict:int = 1, device='cpu'):
    """
    Args:
        max_predict: how many next words to generate
    """
    sent_tokens = tokenizer(given_sentence)
    model.eval()
    for _ in range(max_predict):
        # left padding if token length less than window size
        window = sent_tokens[-seq_len:] if len(sent_tokens) >= seq_len \
                        else ['<pad>'] * (seq_len - len(sent_tokens)) + sent_tokens
        # turn tokens to token_ids, represent in tensor with batch_size=1
        input_ids = torch.tensor([[word2idx.get(w, word2idx['<unk>']) for w in window]], dtype=torch.long).to(device)

        with torch.no_grad():
            logits, attn = model(input_ids)
            next_id = logits.argmax(dim=-1).item()

        next_word = idx2word[next_id]
        sent_tokens.append(next_word) # append to the tail of original tokens for next loop
        print(f"{sent_tokens}")


if __name__ == "__main__":

    # corpus
    sentences = [
        "I drive car to the park with my mother",
        "My mother sits in the car",
        "My sister ride a horse",
        "my mother can not dirve the bus",
        "I go to the park",
        "Jean go to the park",
        "my mother and I go to the restaurant",
        "my brother drive the car to the restaurant",
        "I can ride a horse",
        "my brother drive a bus",
        "Jean can drive the bus",
        "Tom ride a hosre with me",
        "Tom go to the park with me"
    ]
    seq_len = 3   # length of input sequence for training example

    tokenizer = SimpleTokenizer()
    vocabulary = Vocabulary(tokenizer)
    vocab, word2idx, idx2word = vocabulary.build_vocab(sentences)

    # token_ids = vocabulary.get_token_ids(sentences)
    # print(f"{token_ids}")

    dataset = MyDataset(sentences, word2idx)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    torch.manual_seed(42)
    embedding_dim = 4
    qkv_dim = 6
    max_length = max(len(s) for s in sentences)

    attention_model = MyAttentionModel(vocab_size=len(vocabulary), 
                                       embedding_dim=embedding_dim, 
                                       max_seq_len=max_length, 
                                       represent_dim=qkv_dim)

    optimizer = optim.AdamW(attention_model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train model
    train_model(attention_model, loader, loss_fn, optimizer, epochs=100, device=device)

    # eval model
    sample_sentence = "I and tom go to"
    predict_next_n_words(attention_model, sample_sentence, max_predict=2)

    # show heat map of original sentence
    sample_ids, sample_tokens = vocabulary.sentence_to_idx(sample_sentence)
    sample_tensor_ids = torch.tensor([sample_ids])
    positions = torch.arange(len(sample_tokens), device=device).unsqueeze(0)
    with torch.no_grad():
        attn_weights, attn_out = attention_model.cal_attn(sample_tensor_ids, positions)

    utils.plot_attention(attn_weights, sample_tokens)
    

    