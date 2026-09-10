import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Preprocessing =================================================
# dataset from Kaggle: https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus

# combine each word('Word') of a sentence('Sentence #') to a list
# also with corresponding labels('Tag')
def grab_sentences_labels(csv_file_path):
    # load csv dataset from disk
    df = pd.read_csv(csv_file_path, encoding = "ISO-8859-1")
    # print(df.head())

    # forward fill the 'Sentence #' columns in order to group by it
    df["Sentence #"] = df["Sentence #"].ffill()
    sentences = []
    labels = []

    for _, group in df.groupby("Sentence #", sort=False):
        sentences.append(group["Word"].tolist())
        labels.append(group["Tag"].tolist())

    return df, sentences, labels

def build_vocab(sentences):
    word2idx = {}
    idx2word = {}
    for sent in sentences:
        for w in sent:
            if w not in word2idx:
                idx2word[len(word2idx)] = w
                word2idx[w] = len(word2idx)

    unk_idx = len(word2idx)
    pad_idx = unk_idx + 1
    word2idx.update({"<unk>": unk_idx, "<pad>": pad_idx})
    idx2word.update({unk_idx: "<unk>", pad_idx: "<pad>"})
    return word2idx, idx2word

def build_tag(labels):
    tag2idx = {}
    idx2tag = {}

    for labseq in labels:
        for tag in labseq:
            if tag not in tag2idx:
                idx2tag[len(tag2idx)] = tag
                tag2idx[tag] = len(tag2idx)

    return tag2idx, idx2tag

def tokenize_sent_labels(sentences, labels, word2idx, tag2idx):
    X = []
    Y = []
    for sent, lab in zip(sentences, labels):
        x = [word2idx.get(tok, word2idx["<unk>"]) for tok in sent]
        y = [tag2idx[tag] for tag in lab]
        X.append(x)
        Y.append(y)
    return X, Y


# self define dataloader with batches =================================================
def data_loader(batch_size, x, y, pad, shuffle=False):
    '''
    Self defined dataloader with padding, implemented as a generator.
    '''
    
    # total sentences
    num_lines = len(x)
    
    # indicies of sentences that can be shuffled
    lines_index = [*range(num_lines)]
    if shuffle:
        random.shuffle(lines_index)

    # index of indicies which maybe shuffled
    for start in range(0, num_lines, batch_size):
        batch_indices = lines_index[start:start + batch_size]
        buffer_x = [x[index] for index in batch_indices]
        buffer_y = [y[index] for index in batch_indices]
        current_batch_size = len(buffer_x)
        
        # copy from x[index : index + batch_size] 
        # along with corresponding labels y[index : index + batch_size]

        max_len = 0 # the max_len of sentence in this batch
        for sent in buffer_x:
            max_len = max(max_len, len(sent))

        # (batch_size, max_len) 'full' of pad
        X = np.full((current_batch_size, max_len), pad)
        Y = np.full((current_batch_size, max_len), pad)

        # copy from lists to np arrays
        for i in range(current_batch_size):
            X[i,:len(buffer_x[i])] = buffer_x[i]
            Y[i,:len(buffer_y[i])] = buffer_y[i]

        yield((X,Y))


class NamedEntityRecognitionModel(nn.Module):

    def __init__(self, vocab_size, emb_dim, padding_idx, hidden_dim, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_tags)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        logits = self.classifier(out)
        return logits


if __name__ == "__main__":

    # load csv dataset from disk
    csv_file_path = r"E:\PDF\NLP\C3W2\ner_dataset_small.csv"
    _, sentences, labels = grab_sentences_labels(csv_file_path)

    # print(f"sentences: {len(sentences)},  labels: {len(labels)}")

    batch_size = 64

    tag2idx, idx2tag = build_tag(labels=labels)
    w2i, i2w = build_vocab(sentences=sentences)

    x, y = tokenize_sent_labels(
        sentences, labels, word2idx=w2i, tag2idx=tag2idx
    )

    criterion = nn.CrossEntropyLoss(ignore_index=w2i["<pad>"])

    model = NamedEntityRecognitionModel(
        vocab_size=len(w2i),
        emb_dim=64,
        padding_idx=w2i["<pad>"],
        hidden_dim=64,
        num_tags=len(tag2idx),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    epochs = 5
    steps_per_epoch = (len(x) + batch_size - 1) // batch_size
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        data_generator = data_loader(
            batch_size, x, y, w2i["<pad>"], shuffle=True
        )

        for X_batch, Y_batch in data_generator:
            X_batch = torch.from_numpy(X_batch).long()
            Y_batch = torch.from_numpy(Y_batch).long()

            logits = model(X_batch)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                Y_batch.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1:02d}/{epochs}, "
            f"train loss: {total_loss / steps_per_epoch:.4f}"
        )

    # import pprint
    # print(tag2idx)
    # print(idx2tag)
    # print(w2i)
    # print(w2i)
