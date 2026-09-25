"""A small, practical T5-style text-to-text QA example.

This module intentionally keeps the model small enough to run on a laptop while
retaining the important T5 workflow:

* span corruption pre-training targets with sentinel tokens
* a pretrained SentencePiece tokenizer with T5 sentinel tokens
* an encoder-decoder Transformer with tied input/output embeddings
* supervised question-answer training and greedy generation

For production workloads, replace `MiniT5` with `T5ForConditionalGeneration` from Transformers.
"""

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


def sentinel_id(tokenizer: PreTrainedTokenizerBase, index: int) -> int:
	"""Return the ID for a T5 sentinel token such as `<extra_id_0>`."""
	if index < 0:
		raise IndexError("sentinel index must be non-negative")
	return tokenizer.convert_tokens_to_ids(f"<extra_id_{index}>")


def corrupt_spans(
	token_ids: Sequence[int],
	tokenizer: PreTrainedTokenizerBase,
	noise_density: float = 0.15,
	mean_span_length: float = 3.0,
	rng: random.Random | None = None,
) -> tuple[list[int], list[int]]:
	"""Create T5's sentinel-based input and target for span corruption."""
	if not token_ids:
		return [tokenizer.eos_token_id], [tokenizer.eos_token_id]
	if not 0 < noise_density < 1 or mean_span_length <= 0:
		raise ValueError("noise_density must be between 0 and 1 and span length must be positive")

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
		spans = [(0, 1)]
	masked_input: list[int] = []
	target: list[int] = []
	cursor = 0
	for sentinel_index, (start, end) in enumerate(spans):
		masked_input.extend(token_ids[cursor:start])
		sentinel = sentinel_id(tokenizer, sentinel_index)
		masked_input.append(sentinel)
		target.extend((sentinel, *token_ids[start:end]))
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
class ModelConfig:
	d_model: int = 128
	n_heads: int = 2
	num_layers: int = 2
	d_ff: int = 256
	dropout: float = 0.1
	feed_forward_proj: str = "gated-gelu"


class T5RMSNorm(nn.Module):
	def __init__(self, dimension: int, eps: float = 1e-6):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(dimension))
		self.eps = eps

	def forward(self, hidden: Tensor) -> Tensor:
		variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
		return (hidden * torch.rsqrt(variance + self.eps)).type_as(hidden) * self.weight


class T5Attention(nn.Module):
	def __init__(self, config: ModelConfig):
		super().__init__()
		if config.d_model % config.n_heads:
			raise ValueError("d_model must be divisible by n_heads")
		self.attention = nn.MultiheadAttention(
			config.d_model, config.n_heads, config.dropout, batch_first=True, bias=False
		)

	def forward(
		self,
		hidden: Tensor,
		key_value: Tensor | None = None,
		key_padding_mask: Tensor | None = None,
		is_causal: bool = False,
	) -> Tensor:
		key_value = hidden if key_value is None else key_value
		attn_mask = None
		if is_causal:
			length = hidden.size(1)
			attn_mask = torch.triu(
				torch.ones(length, length, dtype=torch.bool, device=hidden.device), diagonal=1
			)
		return self.attention(
			hidden, key_value, key_value, attn_mask=attn_mask,
			key_padding_mask=key_padding_mask, need_weights=False
		)[0]


class T5FeedForward(nn.Module):
	def __init__(self, config: ModelConfig):
		super().__init__()
		self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
		self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
		self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
		self.dropout = nn.Dropout(config.dropout)
		self.gated = config.feed_forward_proj == "gated-gelu"

	def forward(self, hidden: Tensor) -> Tensor:
		if self.gated:
			hidden = F.gelu(self.wi_0(hidden)) * self.wi_1(hidden)
		else:
			hidden = F.relu(self.wi_0(hidden))
		return self.dropout(self.wo(hidden))


class T5Block(nn.Module):
	def __init__(self, config: ModelConfig, is_decoder: bool):
		super().__init__()
		self.is_decoder = is_decoder
		self.self_layer_norm = T5RMSNorm(config.d_model)
		self.self_attention = T5Attention(config)
		if is_decoder:
			self.cross_layer_norm = T5RMSNorm(config.d_model)
			self.cross_attention = T5Attention(config)
		self.ff_layer_norm = T5RMSNorm(config.d_model)
		self.feed_forward = T5FeedForward(config)

	def forward(
		self,
		hidden: Tensor,
		self_padding_mask: Tensor,
		is_causal: bool,
		encoder_hidden: Tensor | None = None,
		cross_padding_mask: Tensor | None = None,
	) -> Tensor:
		hidden = hidden + self.self_attention(
			self.self_layer_norm(hidden), key_padding_mask=self_padding_mask, is_causal=is_causal
		)
		if self.is_decoder and encoder_hidden is not None:
			hidden = hidden + self.cross_attention(
				self.cross_layer_norm(hidden), key_value=encoder_hidden,
				key_padding_mask=cross_padding_mask
			)
		hidden = hidden + self.feed_forward(self.ff_layer_norm(hidden))
		return hidden


class T5Stack(nn.Module):
	def __init__(self, config: ModelConfig, is_decoder: bool):
		super().__init__()
		self.is_decoder = is_decoder
		self.blocks = nn.ModuleList(T5Block(config, is_decoder) for _ in range(config.num_layers))
		self.final_layer_norm = T5RMSNorm(config.d_model)
		self.dropout = nn.Dropout(config.dropout)

	def forward(
		self,
		inputs: Tensor,
		padding_mask: Tensor,
		encoder_hidden: Tensor | None = None,
		cross_padding_mask: Tensor | None = None,
	) -> Tensor:
		hidden = self.dropout(inputs)
		for block in self.blocks:
			hidden = block(
				hidden, padding_mask, self.is_decoder, encoder_hidden, cross_padding_mask
			)
		return self.final_layer_norm(hidden)


class MiniT5(nn.Module):
	"""A compact implementation of T5's shared-embedding architecture."""

	def __init__(self, vocab_size: int, pad_token_id: int, config: ModelConfig):
		super().__init__()
		self.config = config
		self.pad_token_id = pad_token_id
		self.token_embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_token_id)
		self.encoder = T5Stack(config, is_decoder=False)
		self.decoder = T5Stack(config, is_decoder=True)
		self.final_layer = nn.Linear(config.d_model, vocab_size, bias=False)
		self.final_layer.weight = self.token_embedding.weight

	def encode(self, input_ids: Tensor) -> Tensor:
		padding_mask = input_ids.eq(self.pad_token_id)
		return self.encoder(self.token_embedding(input_ids), padding_mask)

	def forward(self, input_ids: Tensor, decoder_input_ids: Tensor) -> Tensor:
		encoder_hidden = self.encode(input_ids)
		decoder_padding = decoder_input_ids.eq(self.pad_token_id)
		decoder_padding[:, 0] = False
		hidden = self.decoder(
			self.token_embedding(decoder_input_ids), decoder_padding, encoder_hidden,
			input_ids.eq(self.pad_token_id)
		)
		return self.final_layer(hidden)

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


def train_qa(model: nn.Module, tokenizer: AutoTokenizer, train_dataloader: DataLoader, optimizer: torch.optim.Optimizer,
			loss_function, epochs: int = 20):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.train()
	for epoch in range(epochs):
		total_loss = 0.0
		for questions, answers in train_dataloader:
			input_ids = pad_sequences(
				[tokenizer.encode(text, add_special_tokens=True) for text in questions], tokenizer.pad_token_id
			).to(device)
			labels = pad_sequences(
				[tokenizer.encode(text, add_special_tokens=True) for text in answers], tokenizer.pad_token_id
			).to(device)
			decoder_input_ids = shift_right(labels, tokenizer.pad_token_id, tokenizer.pad_token_id)
			logits = model(input_ids, decoder_input_ids)
			loss = loss_function(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			total_loss += loss.item()

		if (epoch + 1) % 5 == 0:
			loss = total_loss / max(1, len(train_dataloader))
			print(f"epoch {epoch + 1:>3}/{epochs} - loss: {loss:.4f}")


if __name__ == "__main__":

	seed = 42
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	# we don't build tokenizer and vocabulary from scratch for saving time
	# and focusing on model itself
	tokenizer_name = "t5-small"
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

	# build model with tokenizer and configuration
	model = MiniT5(len(tokenizer), tokenizer.pad_token_id, ModelConfig())
	optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
	loss_function = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

	frame = build_sample_qa()
	train_loader = DataLoader(TextToTextDataset(frame), batch_size=64, shuffle=True)
	train_qa(model, tokenizer, train_loader, optimizer, loss_function, epochs=5)

	# device = next(trained_model.parameters()).device
	# question = "question: what color is the sky? context: the sky is blue."
	# encoded_question = pad_sequences(
	# 	[trained_tokenizer.encode(question, add_special_tokens=True)], trained_tokenizer.pad_token_id
	# ).to(device)
	# answer_ids = trained_model.generate(encoded_question, trained_tokenizer)
	# print(f"question: {question}")
	# print(f"answer: {trained_tokenizer.decode(answer_ids[0].tolist())}")
