import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import re
import os
import random
from pathlib import Path


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


class IMDBTokenizer:

    def __init__(self, max_len, max_vocab_size=10000):
        self.max_len = max_len
        # the target vocabulary size, not exceed
        self.max_vocab_size = max_vocab_size
        # special tokens
        self.word_to_idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.idx_to_word = {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}
        self.word_freq = Counter()

    def __call__(self, text):
        return torch.tensor(self.encode(text), dtype=torch.long)

    def size(self) -> int:
        return len(self.word_to_idx)
    
    def build_vocab(self, directory:str, min_freq=2):
        """build vocabulary"""
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
        for word, freq in most_common:
            if freq >= min_freq:
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                idx += 1
        
        print(f"Vocabulary size: {len(self.word_to_idx)}")


    def tokenize(self, text):
        text = text.lower()
        # remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # remove all punctuations, digits, keep only letters and spaces
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        return words
    
    def encode(self, text):
        words = self.tokenize(text)[:self.max_len-2]  # reserve space for SOS/EOS tokens
        
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
        while len(indices) < self.max_len:
            indices.append(0)
        
        return indices[:self.max_len]

    def decode(self, indicies):
        return [self.idx_to_word[i] for i in indicies]


class IMDBDataset(Dataset):
    def __init__(self, tokenizer, vocab, train=True):
        data_path = os.path.join("E:/PDF/pytorch/C3M3/imdb", "train" if train else "test")
        self.tokenizer = tokenizer
        self.data = []
        self.labels = []
        self.reviews = []

        # load positive
        data_dir = os.path.join(data_path, "pos")
        data_files = os.listdir(data_dir)
        for filename in data_files:
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                self.reviews.append(f.read())
                self.labels.append(1) # pos

        # load negative
        data_dir = os.path.join(data_path, "neg")
        data_files = os.listdir(data_dir)
        for filename in data_files:
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                self.reviews.append(f.read())
                self.labels.append(0) # neg

        # shuffle
        indicies = list(range(len(self.reviews)))
        random.shuffle(indicies)
        self.reviews = [self.reviews[i] for i in indicies]
        self.labels = [self.labels[i] for i in indicies]

        for review, label in zip(self.reviews, self.labels):
            # encode with pad and truncate
            tokens = tokenizer(review)
            indicies = vocab.encode(tokens)
            self.data.append(indicies)
            self.labels.append(label)
        
        # convert to tensors
        self.data = torch.LongTensor(self.data)
        self.labels = torch.LongTensor(self.labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.reviews[idx]



class IMDBSentimentAnalyser(nn.Module):

    def __init__(self, vocab_size, embedding_dim=128, num_layers=2, max_len=512, dropout=0.1):

        super().__init__()
        self.emb_dim = embedding_dim
        
        # word embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # positional embedding
        self.positional_encoding = PositionalEncoding(max_len, embedding_dim)
        # dropout
        self.dropout = nn.Dropout(dropout)
        
        # stack encoder blocks
        self.encoder_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.encoder_layers.append(EncoderBlock(embedding_dim=embedding_dim, nhead=8, ffn_mult=4))
        
        # classifier: pos, neg
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        pos_encoding = self.positional_encoding(x)
        x = x + pos_encoding
        
        x = self.dropout(x)
        
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        
        # pooling, average all tokens per sentence
        x = x.mean(dim=1)  # (batch_size, emb_dim)
        
        output = self.classifier(x)
        return output

def train_encoder(model, dataloader, optimizer, loss_func, lr=0.001, num_epoch=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    for i in range(num_epoch):
        train_total = 0
        train_correct = 0
        train_loss = 0.0
        for reviews, labels, _ in dataloader:
            optimizer.zero_grads()
            reviews = reviews.to(device)
            labels = labels.to(device)
            outputs = model(reviews)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            predicted = torch.argmax(outputs, dim=1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item() * labels.size(0)

        epoch_avg_loss = train_loss / len(dataloader)
        if (i+1) % 10 == 0:
            print(f"Epoch: {i+1}/{num_epoch}   Loss: {epoch_avg_loss}")

def eval_model(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for reviews, labels, _ in dataloader:
            outputs = model(reviews)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f"Accuracy: {(correct / total) * 100:.2f}%")
    

if __name__ == "__main__":

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    tokenizer = IMDBTokenizer(max_len=15)
    tokenizer.build_vocab(directory="E:/PDF/pytorch/C3M3/imdb")
    print(f"total: {tokenizer.size()}")
    test = "I am agree with that, i like this movie a lot."
    print(f"tokens:    {tokenizer.tokenize(test)}")
    encoded_txt = tokenizer.encode(test)
    print(f"encoded:   {encoded_txt}")
    print(f"decoded:   {tokenizer.decode(encoded_txt)}")
    print(f"tensors: {tokenizer(test)}")
    # vocab = IMDBVocabVocabulary(tokenizer)
    
    # train_dataset = IMDBDataset(tokenizer=tokenizer, vocab=vocab, train=True)
    # test_dataset = IMDBDataset(tokenizer=tokenizer, vocab=vocab, train=False)

    """
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    vocab.build_vocab(train_dataloader, test_dataloader)
    """

    # for i in range(10):
    #     _, label, review = test_dataset[i]
    #     print(f"REVIEW: {review}")
        # token_text = vocab.encode(review)
        # print(f"TOKEN: {token_text}")
        # review_text = vocab.decode(token_text)
        # print(f"TURN BACK: {review_text}")
    

    # model = IMDBSentimentAnalyser(vocab.size())
    # loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create a simple encoder block with small dimensions for demonstration
    # encoder_demo = EncoderBlock(embedding_dim=4, nhead=1, ffn_mult=4)


