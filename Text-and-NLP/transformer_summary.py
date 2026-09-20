import math
import random
import re
from pathlib import Path

import pandas as pd
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split


CSV_PATH = Path(__file__).with_name("summary_context.csv")

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
TOKENIZER_PATH = Path(__file__).with_name("summary_tokenizer.json")
TOKENIZER_VOCAB_SIZE = 12_000


class SubwordTokenizer:
    """A compact WordPiece tokenizer trained on the local corpus."""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or Tokenizer(
            models.WordPiece(unk_token=UNK_TOKEN, continuing_subword_prefix="##")
        )
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer.decoder = decoders.WordPiece(prefix="##")

    @classmethod
    def train(cls, texts, vocab_size=TOKENIZER_VOCAB_SIZE):
        tokenizer = cls()
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )
        tokenizer.tokenizer.train_from_iterator(texts, trainer=trainer)
        return tokenizer

    @classmethod
    def from_file(cls, path):
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path):
        self.tokenizer.save(str(path))

    def _special_id(self, token):
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Tokenizer is missing special token: {token}")
        return token_id

    @property
    def pad_id(self):
        return self._special_id(PAD_TOKEN)

    @property
    def unk_id(self):
        return self._special_id(UNK_TOKEN)

    @property
    def bos_id(self):
        return self._special_id(BOS_TOKEN)

    @property
    def eos_id(self):
        return self._special_id(EOS_TOKEN)

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def encode(self, text, max_length, add_bos=False, add_eos=True):
        ids = self.tokenizer.encode(str(text)).ids
        special_count = int(add_bos) + int(add_eos)
        ids = ids[: max(max_length - special_count, 0)]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids.append(self.eos_id)
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
    def collate(batch):
        sources, targets = zip(*batch)
        sources = pad_sequence(list(sources), batch_first=True, padding_value=pad_id)
        targets = pad_sequence(list(targets), batch_first=True, padding_value=pad_id)
        return sources, targets

    return collate


class PositionalEncoding(nn.Module):
    """
    Positional encoding by sin/cos
    with a dropout layer
    """
    def __init__(self, embedding_size, dropout=0.1, max_length=512):
        super().__init__()
        positions = torch.arange(max_length).unsqueeze(1).float()
        frequencies = 1 / (
            10000
            ** (torch.arange(0, embedding_size, 2).float() / embedding_size)
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

    def __init__(self, vocab_size, embedding_size=256, num_heads=8, encoder_layers=3, decoder_layers=3,
                 feed_forward_size=512, dropout=0.1, max_length=512,):
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

    def forward(self, source, decoder_input, pad_id):

        # causal mask，it's a square matrix
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            decoder_input.size(1), device=decoder_input.device
        )
        source_padding = source.eq(pad_id)
        decoder_input_padding = decoder_input.eq(pad_id)

        # The custom tokenizer uses a separate BOS token, so it is never padding.
        decoder_input_padding[:, 0] = False

        source_embedding = self.position(self.shared_embedding(source))
        decoder_input_embedding = self.position(self.shared_embedding(decoder_input))

        # hidden state by Decoder
        hidden = self.transformer(
            source_embedding,
            decoder_input_embedding,
            tgt_mask=causal_mask, # causal mask
            src_key_padding_mask=source_padding,
            tgt_key_padding_mask=decoder_input_padding, # target padding mask
            memory_key_padding_mask=source_padding, # cross attention mask for source padding
        )

        # mapping to vocab by a linear layer
        return self.output(hidden)


def split_decoder_inputs(target, pad_id, bos_id, eos_id):
    """Create decoder inputs and labels for next-token prediction."""
    if target.ndim != 2 or target.size(1) < 2:
        raise ValueError("Target sequences must contain at least BOS and EOS")
    if not torch.all(target[:, 0].eq(bos_id)):
        raise ValueError("Every target sequence must start with BOS")

    decoder_input = target[:, :-1]
    labels = target[:, 1:]
    lengths = target.ne(pad_id).sum(dim=1)
    last_tokens = target.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
    if not torch.all(last_tokens.eq(eos_id)):
        raise ValueError("Every target sequence must end with EOS before padding")
    if decoder_input.shape != labels.shape:
        raise ValueError("Decoder inputs and labels must have the same shape")
    return decoder_input, labels


def run_epoch(model, loader, criterion, pad_id, bos_id, eos_id, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    for source, target in loader:
        source, target = source.to(device), target.to(device)
        decoder_input, labels = split_decoder_inputs(target, pad_id, bos_id, eos_id)

        if training:
            optimizer.zero_grad()

        logits = model(source, decoder_input, pad_id)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite logits detected before calculating loss")
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss detected")

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def train_epoch(model, loader, optimizer, criterion, pad_id, bos_id, eos_id, device):
    return run_epoch(
        model, loader, criterion, pad_id, bos_id, eos_id, device, optimizer
    )


@torch.no_grad()
def validate_epoch(model, loader, criterion, pad_id, bos_id, eos_id, device):
    return run_epoch(model, loader, criterion, pad_id, bos_id, eos_id, device)


@torch.no_grad()
def summarize(model, paragraph, tokenizer, max_source_length, max_target_length, device):
    model.eval()
    source = tokenizer.encode(paragraph, max_source_length, add_bos=False, add_eos=True)
    source = source.unsqueeze(0).to(device)

    generated = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)

    for _ in range(max_target_length - 1):
        logits = model(source, generated, tokenizer.pad_id)
        # select the last token predicted
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
    # shuffle with all data (frac=100%), drop original indicies
    return data.sample(frac=1, random_state=42).reset_index(drop=True) 



if __name__ == "__main__":

    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load csv corpus
    data = load_summary_data(CSV_PATH)

    # Train one tokenizer on both source articles and target summaries.
    tokenizer_texts = pd.concat([data["ctext"], data["text"]]).tolist()
    tokenizer = SubwordTokenizer.train(tokenizer_texts, TOKENIZER_VOCAB_SIZE)
    tokenizer.save(TOKENIZER_PATH)

    dataset = SummaryDataset(data, tokenizer, max_source_length=256, max_target_length=64)

    # split train and valid dataset
    train_size = int(len(dataset) * 0.9)
    valid_size = len(dataset) - train_size
    train_set, valid_set = random_split(
        dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42)
    )

    collate = make_collate_fn(tokenizer.pad_id)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False, collate_fn=collate)

    model = SummarizationTransformer(
        vocab_size=tokenizer.vocab_size,
        embedding_size=128,
        num_heads=2,
        encoder_layers=2,
        decoder_layers=2,
        feed_forward_size=256, 
        dropout=0.1,
        max_length=256,
    ).to(device)

    # training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    epochs = 30
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    print(f"device={device}, examples={len(dataset)}")
    print(f"subword vocabulary size={tokenizer.vocab_size}")

    for epoch in range(epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            tokenizer.pad_id,
            tokenizer.bos_id,
            tokenizer.eos_id,
            device,
        )
        valid_loss = validate_epoch(
            model,
            valid_loader,
            criterion,
            tokenizer.pad_id,
            tokenizer.bos_id,
            tokenizer.eos_id,
            device,
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch + 1:02d} | train loss {train_loss:.4f} "
            f"| valid loss {valid_loss:.4f} | lr {current_lr:.2e}"
        )

    # evaluate with a single sample
    example = data.iloc[0]["ctext"]
    print("\nsource:", example)
    print("\nsummary:", summarize(model, example, tokenizer, 256, 64, device))