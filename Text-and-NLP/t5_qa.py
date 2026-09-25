"""Fine-tune a small pretrained T5 model for question answering."""

import random

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset


class TextToTextDataset(Dataset[dict[str, str]]):
	def __init__(self, frame: pd.DataFrame):
		required = {"input_text", "target_text"}
		if not required.issubset(frame.columns):
			raise ValueError(f"dataset must contain columns: {sorted(required)}")
		self.examples = list(frame[["input_text", "target_text"]].itertuples(index=False, name=None))

	def __len__(self) -> int:
		return len(self.examples)

	def __getitem__(self, index: int) -> dict[str, str]:
		input_text, target_text = self.examples[index]
		return {"input_text": input_text, "target_text": target_text}


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


def train_qa(
	model: T5ForConditionalGeneration,
	train_dataloader: DataLoader,
	optimizer: torch.optim.Optimizer,
	epochs: int = 20,
):
	"""Fine-tune pretrained T5 with teacher forcing and its built-in loss."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.train()
	for epoch in range(epochs):
		total_loss = 0.0
		for batch in train_dataloader:
			batch = {name: value.to(device) for name, value in batch.items()}
			loss = model(**batch).loss
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

	# t5-small is a pretrained 60M-parameter T5 checkpoint.
	tokenizer_name = "google-t5/t5-small"
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
	model = T5ForConditionalGeneration.from_pretrained(tokenizer_name)
	optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

	frame = build_sample_qa()
	dataset = TextToTextDataset(frame)
	train_loader = DataLoader(
		dataset, batch_size=4, shuffle=True,
		collate_fn=make_collate_fn(tokenizer, model),
	)
	train_qa(model, train_loader, optimizer, epochs=5)

	model.eval()
	question = "question: what color is the sky? context: the sky is blue."
	inputs = tokenizer(question, return_tensors="pt").to(model.device)
	answer_ids = model.generate(**inputs, max_new_tokens=16)
	print(f"question: {question}")
	print(f"answer: {tokenizer.decode(answer_ids[0], skip_special_tokens=True)}")
