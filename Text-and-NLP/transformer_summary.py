import math
import random
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split


CSV_PATH = Path(__file__).with_name("summary_context.csv")

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


class HuggingFaceTokenizer:
    """Pretrained tokenizer shared by the encoder and decoder."""

    def __init__(self, model_name="t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pad_id = self.tokenizer.pad_token_id
        self.unk_id = self.tokenizer.unk_token_id
        self.eos_id = self.tokenizer.eos_token_id
        # T5 uses the padding token as the decoder start token.
        self.bos_id = self.tokenizer.pad_token_id

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def encode(self, text, max_length, add_bos=False, add_eos=True):
        available = max_length - int(add_bos)
        encoded = self.tokenizer(
            str(text),
            add_special_tokens=add_eos,
            max_length=max(available, 0),
            truncation=True,
            return_tensors="pt",
        )
        ids = encoded["input_ids"][0].tolist()
        if add_bos:
            ids = [self.bos_id] + ids

        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()


class SummaryDataset(Dataset):
    def __init__(self, data, tokenizer, max_source_length, max_target_length):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        source = self.tokenizer.encode(
            row["ctext"], self.max_source_length, add_bos=False, add_eos=True
        )
        target = self.tokenizer.encode(
            row["text"], self.max_target_length, add_bos=True, add_eos=True
        )
        return source, target


def make_collate_fn(pad_id):
    def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]):
        sources, targets = zip(*batch)
        sources = pad_sequence(list(sources), batch_first=True, padding_value=pad_id)
        targets = pad_sequence(list(targets), batch_first=True, padding_value=pad_id)
        return sources, targets

    return collate


class PositionalEncoding(nn.Module):
    """
    Positional encoding by sin/cos
    """
    def __init__(self, embedding_size, dropout=0.1, max_length=512):
        super().__init__()
        positions = torch.arange(max_length).unsqueeze(1).float()
        frequencies = torch.exp(
            torch.arange(0, embedding_size, 2).float()
            * (-math.log(10000.0) / embedding_size)
        )
        encoding = torch.zeros(max_length, embedding_size)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        # defensive slicing for even positions
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        # register as parameter
        self.register_buffer("encoding", encoding.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings):
        return self.dropout(embeddings + self.encoding[:, : embeddings.size(1)])


class SummarizationTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
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
        self.shared_embedding = nn.Embedding(vocab_size, embedding_size)
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
        self.output = nn.Linear(embedding_size, vocab_size)

    def forward(self, source, target, pad_id):
        target_input = target[:, :-1]
        target_mask = nn.Transformer.generate_square_subsequent_mask(
            target_input.size(1), device=target.device
        )
        source_padding = source.eq(pad_id)
        target_padding = target_input.eq(pad_id)

        source_embedding = self.position(
            self.shared_embedding(source) * math.sqrt(self.embedding_size)
        )
        target_embedding = self.position(
            self.shared_embedding(target_input) * math.sqrt(self.embedding_size)
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


def train_epoch(model, loader, optimizer, criterion, pad_id, device):
    model.train()
    total_loss = 0.0
    for source, target in loader:
        source, target = source.to(device), target.to(device)
        optimizer.zero_grad()

        logits = model(source, target, pad_id)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target[:, 1:].reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def summarize(model, paragraph, tokenizer, max_source_length, max_target_length, device):
    model.eval()
    source = tokenizer.encode(paragraph, max_source_length, add_bos=False, add_eos=True)
    source = source.unsqueeze(0).to(device)

    generated = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)

    for _ in range(max_target_length - 1):
        logits = model(source, generated, tokenizer.pad_id)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if next_token.item() == tokenizer.eos_id:
            break

    return tokenizer.decode(generated[0].tolist()[1:])


def load_summary_data(csv_path):
    data = pd.read_csv(csv_path)[["text", "ctext"]].dropna()
    data["text"] = data["text"].astype(str).str.strip()
    data["ctext"] = data["ctext"].astype(str).str.strip()
    data = data[(data["text"] != "") & (data["ctext"] != "")].drop_duplicates()
    return data.sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load csv corpus
    data = load_summary_data(CSV_PATH)

    # using `t5-small` tokenizer for processing
    tokenizer = HuggingFaceTokenizer("t5-small")

    dataset = SummaryDataset(data, tokenizer, max_source_length=256, max_target_length=64)

    # split train and valid dataset
    train_size = int(len(dataset) * 0.9)
    valid_size = len(dataset) - train_size
    train_set, valid_set = random_split(
        dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42)
    )

    collate = make_collate_fn(tokenizer.pad_id)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate)

    model = SummarizationTransformer(
        vocab_size=tokenizer.vocab_size,
        embedding_size=256,
        max_length=256,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    print(f"device={device}, examples={len(dataset)}")
    print(f"shared vocabulary size={tokenizer.vocab_size}")

    for epoch in range(10):
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            tokenizer.pad_id,
            device,
        )
        print(f"epoch {epoch + 1:02d} | loss {loss:.4f}")

    example = data.iloc[0]["ctext"]
    print("\nsource:", example)
    print("summary:", summarize(model, example, tokenizer, 256, 64, device))


if __name__ == "__main__":
    main()