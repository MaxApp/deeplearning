"""Fine-tune a small pretrained T5 model for question answering."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, T5ForConditionalGeneration
from torch.utils.data import DataLoader, IterableDataset

SQUAD_FILE = "Squad_v2.0.json"

class TextToTextDataset(IterableDataset[dict[str, str]]):
	"""Stream SQuAD v2 examples from its downloaded JSON file."""

	def __init__(self, json_path: str, shuffle_buffer_size: int = 1_000, seed: int = 42):
		super().__init__()
		self.json_path = Path(json_path)
		if not self.json_path.is_file():
			raise FileNotFoundError(f"SQuAD JSON file not found: {self.json_path}")
		
		self.dataset = load_dataset(
			"json",
			data_files=str(self.json_path),
			field="data",
			split="train",
			streaming=True,
		)
		self.dataset = self.dataset.map(
			self._flatten_batch,
			batched=True,
			batch_size=1,
			remove_columns=self.dataset.column_names,
		)
		if shuffle_buffer_size > 0:
			self.dataset = self.dataset.shuffle(
				buffer_size=shuffle_buffer_size,
				seed=seed,
			)

	@staticmethod
	def _flatten_batch(batch: dict) -> dict[str, list[str]]:
		input_texts = []
		target_texts = []
		for paragraphs in batch["paragraphs"]:
			for paragraph in paragraphs:
				context = paragraph["context"]
				for qa in paragraph["qas"]:
					answers = qa.get("answers", [])
					input_texts.append(f"question: {qa['question']} context: {context}")
					target_texts.append(
						"unanswerable"
						if qa.get("is_impossible", False) or not answers
						else answers[0]["text"]
					)
		return {"input_text": input_texts, "target_text": target_texts}

	def __iter__(self):
		yield from self.dataset

	def set_epoch(self, epoch: int) -> None:
		self.dataset.set_epoch(epoch)


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


def make_collate_fn(tokenizer: AutoTokenizer, model: T5ForConditionalGeneration):
	"""Tokenize text-to-text examples and let Transformers pad each batch."""
	collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="pt")

	def collate(examples: list[dict[str, str]]) -> dict[str, torch.Tensor]:
		features = []
		for example in examples:
			model_inputs = tokenizer(example["input_text"], truncation=True, max_length=128)
			labels = tokenizer(text_target=example["target_text"], truncation=True, max_length=64)
			model_inputs["labels"] = labels["input_ids"]
			features.append(model_inputs)
		return collator(features)

	return collate


def train_qa(model: T5ForConditionalGeneration,
			train_dataloader: DataLoader,
			optimizer: torch.optim.Optimizer,
			epochs: int = 20):

	"""Fine-tune pretrained T5 with teacher forcing and its built-in loss."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.train()
	for epoch in range(epochs):
		set_epoch = getattr(train_dataloader.dataset, "set_epoch", None)
		if set_epoch is not None:
			set_epoch(epoch)
		total_loss = 0.0
		batch_count = 0
		for batch in train_dataloader:
			batch = {name: value.to(device) for name, value in batch.items()}
			loss = model(**batch).loss
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			total_loss += loss.item()
			batch_count += 1

		if (epoch + 1) % 5 == 0:
			loss = total_loss / max(1, batch_count)
			print(f"epoch {epoch + 1:>3}/{epochs} - loss: {loss:.4f}")


if __name__ == "__main__":

	seed = 42
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	# t5-small is a pretrained 60M-parameter T5 checkpoint.
	tokenizer_name = "google-t5/t5-small"
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
	model = T5ForConditionalGeneration.from_pretrained(tokenizer_name)
	optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

	dataset_path = Path(__file__).with_name("train-v2.0.json")
	dataset = TextToTextDataset(dataset_path, shuffle_buffer_size=1_000, seed=seed)
	train_loader = DataLoader(
		dataset, batch_size=4,
		collate_fn=make_collate_fn(tokenizer, model),
	)
	train_qa(model, train_loader, optimizer, epochs=5)

	model.eval()
	question = "question: what color is the sky? context: the sky is blue."
	inputs = tokenizer(question, return_tensors="pt").to(model.device)
	answer_ids = model.generate(**inputs, max_new_tokens=16)
	print(f"question: {question}")
	print(f"answer: {tokenizer.decode(answer_ids[0], skip_special_tokens=True)}")
