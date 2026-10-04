"""Fine-tune a small pretrained T5 model for question answering."""

from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset

SQUAD_FILE = Path(__file__).resolve().with_name("Squad_v2.0.json")
SEED = 42


class TextToTextDataset(Dataset[dict[str, str]]):
	"""Load flattened SQuAD v2 examples from a JSON file."""

	def __init__(self, json_path: str | Path):
		super().__init__()
		self.json_path = Path(json_path)
		if not self.json_path.is_file():
			raise FileNotFoundError(f"SQuAD JSON file not found: {self.json_path}")

		self.dataset = load_dataset(
			"json",
			data_files=str(self.json_path),
			field="data",
			split="train",
			# streaming=True,
		)
		self.dataset = self.dataset.map(
			self._flatten_batch,
			batched=True,
			batch_size=1,
			remove_columns=self.dataset.column_names,
		)

	@staticmethod
	def _flatten_batch(batch: dict) -> dict[str, list[str]]:
		"""
		Return examples in the format:
			input_texts: ["question: xxx context: xxx"]
			target_texts: ["answer text"] or ["unanswerable"]
		"""
		input_texts = []
		target_texts = []
		for paragraphs in batch["paragraphs"]:
			for paragraph in paragraphs:
				context = paragraph["context"]
				for qa in paragraph["qas"]:
					answers = qa.get("answers", [])
					input_texts.append(f"question: {qa['question']} context: {context}")
					if not qa.get("is_impossible", False) and answers:
						target_texts.append(answers[0]['text'])
					else:
						target_texts.append("unanswerable")
		return {"input_texts": input_texts, "target_texts": target_texts}

	def __len__(self) -> int:
		return len(self.dataset)

	def __getitem__(self, index: int) -> dict[str, str]:
		return self.dataset[index]


def make_collate_fn(tokenizer: AutoTokenizer, model: T5ForConditionalGeneration):
	"""Tokenize text-to-text examples and let Transformers pad each batch."""
	# It can use T5’s prepare_decoder_input_ids_from_labels to build decoder_input_ids when it batches the labels. 
	# If they aren’t supplied, T5 can also prepare them internally when called with labels. 
	collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="pt")

	def collate(examples: list[dict[str, str]]) -> dict[str, torch.Tensor]:
		features = []
		for example in examples:
			# the tokenizer returns dictionary like:
			# {
			# 	"input_ids":      [token_id, token_id, ...],
			# 	"attention_mask": [1, 1, ...],
			# }
			# 
			# so the `model_inputs` already contains `attention_mask` as inputs mask id
			model_inputs = tokenizer(example["input_texts"], truncation=True, max_length=256)
			labels = tokenizer(text_target=example["target_texts"], truncation=True, max_length=64)
			model_inputs["labels"] = labels["input_ids"]
			features.append(model_inputs)

		# The collator then pads labels to the longest target in the batch, using -100 by default. 
		# T5’s loss ignores positions with label -100.
		# That applies to target labels, not the model’s input IDs. 
		# Input padding remains the tokenizer’s pad token and is masked by attention_mask.
		return collator(features)

	return collate


def train_qa(model: T5ForConditionalGeneration, train_dataloader: DataLoader, optimizer: torch.optim.Optimizer, epochs: int = 10):

	"""Fine-tune pretrained T5 with teacher forcing and its built-in loss"""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.train()
	for epoch in range(epochs):
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

		if batch_count == 0:
			raise RuntimeError(f"Training epoch {epoch + 1} produced no batches.")
		if (epoch + 1) % 5 == 0:
			loss = total_loss / batch_count
			print(f"epoch {epoch + 1:>3}/{epochs} loss: {loss:.4f}")


if __name__ == "__main__":
	torch.manual_seed(SEED)

	# using pretrained `t5-small` checkpoint for Q&A
	tokenizer_name = "google-t5/t5-small"
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
	model = T5ForConditionalGeneration.from_pretrained(tokenizer_name)
	optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

	dataset = TextToTextDataset(SQUAD_FILE)
	shuffle_generator = torch.Generator().manual_seed(SEED)
	train_loader = DataLoader(
		dataset,
		batch_size=4,
		shuffle=True,
		generator=shuffle_generator,
		collate_fn=make_collate_fn(tokenizer, model),
	)
	train_qa(model, train_loader, optimizer, epochs=5)
