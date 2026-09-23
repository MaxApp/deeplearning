"""A small, practical T5-style text-to-text QA example.

This module intentionally keeps the model small enough to run on a laptop while
retaining the important T5 workflow:

* span corruption pre-training targets with sentinel tokens
* a pretrained SentencePiece tokenizer with T5 sentinel tokens
* an encoder-decoder Transformer with tied input/output embeddings
* supervised question-answer training and greedy generation

For production workloads, replace :class:`MiniT5` with
``T5ForConditionalGeneration`` from Transformers. The data preparation and
training interfaces are deliberately similar.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


def sentinel_id(tokenizer: PreTrainedTokenizerBase, index: int) -> int:
	"""Return the ID for a T5 sentinel token such as ``<extra_id_0>``."""
	if index < 0:
		raise IndexError("sentinel index must be non-negative")
	return tokenizer.convert_tokens_to_ids(f"<extra_id_{index}>")


def corrupt_spans(
	token_ids: Sequence[int],
	tokenizer: PreTrainedTokenizerBase,
	*,
	noise_density: float = 0.15,
	mean_span_length: float = 3.0,
	rng: random.Random | None = None,
) -> tuple[list[int], list[int]]:
	"""Create T5's input and target for span corruption.

	Every masked span is replaced by one sentinel in the input. The target
	contains the same sentinel followed by the removed tokens, ending in EOS.
	"""
	if not token_ids:
		return [tokenizer.eos_token_id], [tokenizer.eos_token_id]
	if not 0 < noise_density < 1:
		raise ValueError("noise_density must be between 0 and 1")

	random_source = rng or random.Random()
	length = len(token_ids)
	num_noise = max(1, min(length - 1 if length > 1 else 1, round(length * noise_density)))
	num_spans = max(1, round(num_noise / mean_span_length))
	starts = sorted(random_source.sample(range(length), min(num_spans, length)))
	spans: list[tuple[int, int]] = []
	remaining = num_noise
	for index, start in enumerate(starts):
		end_limit = starts[index + 1] if index + 1 < len(starts) else length
		span_length = max(1, round(random_source.expovariate(1 / mean_span_length)))
		end = min(end_limit, start + span_length, start + remaining)
		if end > start:
			spans.append((start, end))
			remaining -= end - start
		if remaining <= 0:
			break

	if not spans:
		spans = [(0, min(1, length))]

	masked_input: list[int] = []
	target: list[int] = []
	cursor = 0
	for sentinel_index, (start, end) in enumerate(spans):
		masked_input.extend(token_ids[cursor:start])
		sentinel = sentinel_id(tokenizer, sentinel_index)
		masked_input.append(sentinel)
		target.append(sentinel)
		target.extend(token_ids[start:end])
		cursor = end
	masked_input.extend(token_ids[cursor:])
	target.append(tokenizer.eos_token_id)
	return masked_input, target


def shift_right(labels: Tensor, decoder_start_token_id: int, pad_token_id: int) -> Tensor:
	"""Build teacher-forcing decoder inputs from target labels."""
	decoder_input_ids = labels.new_full(labels.shape, pad_token_id)
	decoder_input_ids[:, 0] = decoder_start_token_id
	decoder_input_ids[:, 1:] = labels[:, :-1]
	return decoder_input_ids


def pad_sequences(sequences: Sequence[Sequence[int]], pad_token_id: int) -> Tensor:
	width = max(len(sequence) for sequence in sequences)
	batch = [list(sequence) + [pad_token_id] * (width - len(sequence)) for sequence in sequences]
	return torch.tensor(batch, dtype=torch.long)


class TextToTextDataset(Dataset[tuple[str, str]]):
	def __init__(self, frame: pd.DataFrame):
		required = {"input_text", "target_text"}
		if not required.issubset(frame.columns):
			raise ValueError(f"dataset must contain columns: {sorted(required)}")
		self.examples = list(frame[["input_text", "target_text"]].itertuples(index=False, name=None))

	def __len__(self) -> int:
		return len(self.examples)

	def __getitem__(self, index: int) -> tuple[str, str]:
		return self.examples[index]


@dataclass
class MiniT5Config:
	d_model: int = 128
	n_heads: int = 4
	num_layers: int = 2
	d_ff: int = 256
	dropout: float = 0.1
	max_length: int = 96


class MiniT5(nn.Module):
	"""A compact T5-style encoder-decoder Transformer for experiments."""

	def __init__(self, vocab_size: int, config: MiniT5Config, pad_token_id: int):
		super().__init__()
		self.config = config
		self.pad_token_id = pad_token_id
		self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_token_id)
		self.position_embeddings = nn.Embedding(config.max_length, config.d_model)
		encoder_layer = nn.TransformerEncoderLayer(
			config.d_model, config.n_heads, config.d_ff, config.dropout, batch_first=True, norm_first=True
		)
		decoder_layer = nn.TransformerDecoderLayer(
			config.d_model, config.n_heads, config.d_ff, config.dropout, batch_first=True, norm_first=True
		)
		self.encoder = nn.TransformerEncoder(encoder_layer, config.num_layers)
		self.decoder = nn.TransformerDecoder(decoder_layer, config.num_layers)
		self.final_layer_norm = nn.LayerNorm(config.d_model)
		self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)
		self.lm_head.weight = self.shared.weight
		self.dropout = nn.Dropout(config.dropout)

	def _embed(self, input_ids: Tensor) -> Tensor:
		positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
		return self.dropout(self.shared(input_ids) * math.sqrt(self.config.d_model) + self.position_embeddings(positions))

	def encode(self, input_ids: Tensor) -> Tensor:
		padding_mask = input_ids.eq(self.pad_token_id)
		return self.encoder(self._embed(input_ids), src_key_padding_mask=padding_mask)

	def forward(self, input_ids: Tensor, decoder_input_ids: Tensor) -> Tensor:
		memory = self.encode(input_ids)
		decoder_padding_mask = decoder_input_ids.eq(self.pad_token_id)
		causal_mask = nn.Transformer.generate_square_subsequent_mask(
			decoder_input_ids.size(1), device=decoder_input_ids.device
		)
		hidden = self.decoder(
			self._embed(decoder_input_ids), memory, tgt_mask=causal_mask,
			tgt_key_padding_mask=decoder_padding_mask, memory_key_padding_mask=input_ids.eq(self.pad_token_id)
		)
		return self.lm_head(self.final_layer_norm(hidden))

	@torch.no_grad()
	def generate(self, input_ids: Tensor, tokenizer: PreTrainedTokenizerBase, max_new_tokens: int = 32) -> Tensor:
		self.eval()
		generated = input_ids.new_full((input_ids.size(0), 1), tokenizer.pad_token_id)
		for _ in range(max_new_tokens):
			logits = self(input_ids, generated)
			next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
			generated = torch.cat((generated, next_token), dim=1)
			if bool(next_token.eq(tokenizer.eos_token_id).all()):
				break
		return generated[:, 1:]


def train_epoch(
	model: MiniT5,
	loader: DataLoader[tuple[str, str]],
	tokenizer: PreTrainedTokenizerBase,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
) -> float:
	model.train()
	total_loss = 0.0
	loss_function = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
	for questions, answers in loader:
		input_ids = pad_sequences(
			[tokenizer.encode(text, add_special_tokens=True) for text in questions], tokenizer.pad_token_id
		).to(device)
		labels = pad_sequences(
			[tokenizer.encode(text, add_special_tokens=True) for text in answers], tokenizer.pad_token_id
		).to(device)
		decoder_input_ids = shift_right(labels, tokenizer.pad_token_id, tokenizer.pad_token_id)
		logits = model(input_ids, decoder_input_ids)
		loss = loss_function(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()
		total_loss += loss.item()
	return total_loss / max(1, len(loader))


def build_sample_qa() -> pd.DataFrame:
	"""Return a tiny dataset suitable for a smoke test or classroom demo."""
	return pd.DataFrame(
		{
			"input_text": [
				"question: what color is the sky? context: the sky is blue.",
				"question: what animal barks? context: a dog barks loudly.",
				"question: where do fish live? context: fish live in water.",
				"question: what do bees make? context: bees make honey.",
			],
			"target_text": ["blue", "dog", "water", "honey"],
		}
	)


def train_qa(
	frame: pd.DataFrame | None = None,
	*,
	epochs: int = 20,
	batch_size: int = 4,
	seed: int = 7,
) -> tuple[MiniT5, PreTrainedTokenizerBase]:
	"""Build and train a small QA model, returning both model and tokenizer."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	frame = build_sample_qa() if frame is None else frame.copy()
	tokenizer = AutoTokenizer.from_pretrained("t5-small")
	model = MiniT5(len(tokenizer), MiniT5Config(), tokenizer.pad_token_id)
	loader = DataLoader(TextToTextDataset(frame), batch_size=batch_size, shuffle=True)
	optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	for epoch in range(epochs):
		loss = train_epoch(model, loader, tokenizer, optimizer, device)
		if (epoch + 1) % max(1, epochs // 4) == 0:
			print(f"epoch {epoch + 1:>3}/{epochs} - loss: {loss:.4f}")
	return model, tokenizer


if __name__ == "__main__":
	trained_model, trained_tokenizer = train_qa()
	device = next(trained_model.parameters()).device
	question = "question: what color is the sky? context: the sky is blue."
	encoded_question = pad_sequences(
		[trained_tokenizer.encode(question, add_special_tokens=True)], trained_tokenizer.pad_token_id
	).to(device)
	answer_ids = trained_model.generate(encoded_question, trained_tokenizer)
	print(f"question: {question}")
	print(f"answer: {trained_tokenizer.decode(answer_ids[0].tolist())}")
