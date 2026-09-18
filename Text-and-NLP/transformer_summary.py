import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split


CSV_PATH = Path(__file__).with_name("summary_context.csv")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


def tokenize(text):
    """Keep words and punctuation as separate tokens."""
    return re.findall(r"\w+|[^\w\s]", str(text).lower(), flags=re.UNICODE)


class Vocabulary:
    def __init__(self, texts, min_frequency=2):
        counts = Counter(token for text in texts for token in tokenize(text))
        words = sorted(
            token for token, count in counts.items() if count >= min_frequency
        )
        self.itos = SPECIAL_TOKENS + words
        self.stoi = {token: index for index, token in enumerate(self.itos)}

        self.pad_id = self.stoi[PAD_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]
        self.bos_id = self.stoi[BOS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]

    def encode(self, text, max_length, add_bos=False, add_eos=True):
        tokens = tokenize(text)[: max_length - int(add_bos) - int(add_eos)]
        ids = [self.stoi.get(token, self.unk_id) for token in tokens]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        words = []
        for index in ids:
            token = self.itos[int(index)]
            if token == EOS_TOKEN:
                break
            if token not in SPECIAL_TOKENS:
                words.append(token)

        text = " ".join(words)
        return re.sub(r"\s+([,.!?;:])", r"\1", text)


class SummaryDataset(Dataset):
    def __init__(self, data, source_vocab, target_vocab, max_source_length, max_target_length):
        self.data = data.reset_index(drop=True)
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        source = self.source_vocab.encode(
            row["ctext"], self.max_source_length, add_eos=True
        )
        target = self.target_vocab.encode(
            row["text"], self.max_target_length, add_bos=True, add_eos=True
        )
        return source, target


def make_collate_fn(source_pad_id, target_pad_id):
    def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]):
        sources, targets = zip(*batch)
        sources = pad_sequence(list(sources), batch_first=True, padding_value=source_pad_id)
        targets = pad_sequence(list(targets), batch_first=True, padding_value=target_pad_id)
        return sources, targets

    return collate


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout=0.1, max_length=512):
        super().__init__()
        positions = torch.arange(max_length).unsqueeze(1).float()
        frequencies = torch.exp(
            torch.arange(0, embedding_size, 2).float()
            * (-math.log(10000.0) / embedding_size)
        )
        encoding = torch.zeros(max_length, embedding_size)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings):
        encoding = cast(torch.Tensor, self.encoding)
        return self.dropout(embeddings + encoding[:, : embeddings.size(1)])


class SummarizationTransformer(nn.Module):
    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        embedding_size=256,
        num_heads=8,
        encoder_layers=3,
        decoder_layers=3,
        feed_forward_size=512,
        dropout=0.1,
        max_length=512,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.source_embedding = nn.Embedding(source_vocab_size, embedding_size)
        self.target_embedding = nn.Embedding(target_vocab_size, embedding_size)
        self.position = PositionalEncoding(embedding_size, dropout, max_length)
        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers,
            dim_feedforward=feed_forward_size,
            dropout=dropout,
            batch_first=True,
        )
        self.output = nn.Linear(embedding_size, target_vocab_size)

    def forward(self, source, target, source_pad_id, target_pad_id):
        target_input = target[:, :-1]
        target_mask = nn.Transformer.generate_square_subsequent_mask(
            target_input.size(1), device=target.device
        )
        source_padding = source.eq(source_pad_id)
        target_padding = target_input.eq(target_pad_id)

        source_embedding = self.position(
            self.source_embedding(source) * math.sqrt(self.embedding_size)
        )
        target_embedding = self.position(
            self.target_embedding(target_input) * math.sqrt(self.embedding_size)
        )
        hidden = self.transformer(
            source_embedding,
            target_embedding,
            tgt_mask=target_mask,
            src_key_padding_mask=source_padding,
            tgt_key_padding_mask=target_padding,
            memory_key_padding_mask=source_padding,
        )
        return self.output(hidden)


def train_epoch(model, loader, optimizer, criterion, source_pad_id, target_pad_id, device):
    model.train()
    total_loss = 0.0
    for source, target in loader:
        source, target = source.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(source, target, source_pad_id, target_pad_id)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def summarize(model, paragraph, source_vocab, target_vocab, max_source_length, max_target_length, device):
    model.eval()
    source = source_vocab.encode(paragraph, max_source_length, add_eos=True)
    source = source.unsqueeze(0).to(device)
    generated = torch.tensor([[target_vocab.bos_id]], dtype=torch.long, device=device)

    for _ in range(max_target_length - 1):
        logits = model(source, generated, source_vocab.pad_id, target_vocab.pad_id)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == target_vocab.eos_id:
            break
    return target_vocab.decode(generated[0].tolist()[1:])


def main():
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ctext is the full paragraph; text is the human-written summary.
    data = pd.read_csv(CSV_PATH)[["text", "ctext"]].dropna()
    data["text"] = data["text"].astype(str).str.strip()
    data["ctext"] = data["ctext"].astype(str).str.strip()
    data = data[(data["text"] != "") & (data["ctext"] != "")].drop_duplicates()
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    source_vocab = Vocabulary(data["ctext"], min_frequency=2)
    target_vocab = Vocabulary(data["text"], min_frequency=1)
    dataset = SummaryDataset(data, source_vocab, target_vocab, 256, 64)
    train_size = max(1, int(len(dataset) * 0.9))
    valid_size = len(dataset) - train_size
    train_set, valid_set = random_split(
        dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42)
    )
    collate = make_collate_fn(source_vocab.pad_id, target_vocab.pad_id)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate)

    model = SummarizationTransformer(
        len(source_vocab.itos), len(target_vocab.itos), max_length=256
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=target_vocab.pad_id)

    print(f"device={device}, examples={len(dataset)}")
    print(f"source vocabulary={len(source_vocab.itos)}, target vocabulary={len(target_vocab.itos)}")
    for epoch in range(10):
        loss = train_epoch(
            model, train_loader, optimizer, criterion,
            source_vocab.pad_id, target_vocab.pad_id, device,
        )
        print(f"epoch {epoch + 1:02d} | loss {loss:.4f}")

    example = data.iloc[0]["ctext"]
    print("\nsource:", example)
    print("summary:", summarize(
        model, example, source_vocab, target_vocab, 256, 64, device
    ))


if __name__ == "__main__":
    main()
