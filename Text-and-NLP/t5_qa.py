"""Fine-tune a small pretrained T5 model for question answering."""

from pathlib import Path

import torch
from datasets import Dataset as HFDataset, load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset

SQUAD_FILE = "Squad_v2.0.json"
SEED = 42


class TextToTextDataset(Dataset[dict[str, str | int]]):
	"""Flatten a SQuAD v2 article split into question/context/target examples."""

	def __init__(self, records: HFDataset):
		super().__init__()
		self.dataset = records.map(
			self._flatten_batch,
			batched=True,
			batch_size=1,
			remove_columns=records.column_names,
		)

	@staticmethod
	def _flatten_batch(batch: dict) -> dict[str, list]:
		"""
		Return examples in the format:
			question_texts: ["xxx"]
			context_texts: ["xxx"]
			target_texts: ["answer text"] or ["unanswerable"]
			answer_starts: [character offset] or [-1]
		"""
		question_texts = []
		context_texts = []
		target_texts = []
		answer_starts = []
		for paragraphs in batch["paragraphs"]:
			for paragraph in paragraphs:
				context = paragraph["context"]
				for qa in paragraph["qas"]:
					answers = qa.get("answers", [])
					question_texts.append(qa["question"])
					context_texts.append(context)
					if not qa.get("is_impossible", False) and answers:
						target_texts.append(answers[0]["text"])
						answer_starts.append(int(answers[0]["answer_start"]))
					else:
						target_texts.append("unanswerable")
						answer_starts.append(-1)
		return {
			"question_texts": question_texts,
			"context_texts": context_texts,
			"target_texts": target_texts,
			"answer_starts": answer_starts,
		}

	def __len__(self) -> int:
		return len(self.dataset)

	def __getitem__(self, index: int) -> dict[str, str | int]:
		return self.dataset[index]


def make_collate_fn(tokenizer: AutoTokenizer, model: T5ForConditionalGeneration):
	"""Tokenize examples and crop long contexts around answer spans."""
	max_source_length = 512
	max_question_length = 128
	collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="pt")

	def collate(examples: list[dict]) -> dict[str, torch.Tensor]:
		question_tokens = tokenizer(
			[f"question: {example['question_texts']} context:" for example in examples],
			add_special_tokens=False,
			truncation=True,
			max_length=max_question_length,
		)["input_ids"]

		context_tokens = tokenizer(
			[example["context_texts"] for example in examples],
			add_special_tokens=False,
			return_offsets_mapping=True, # return each context token position info
		)

		target_tokens = tokenizer(
			text_target=[example["target_texts"] for example in examples],
			truncation=True,
			max_length=64,
		)["input_ids"]

		features = []
		for example, question_ids, context_ids, offsets, labels in zip(
			examples,
			question_tokens,
			context_tokens["input_ids"],
			context_tokens["offset_mapping"],
			target_tokens,
		):
			answer_start = example["answer_starts"]
			context_window_start = 0
			context_budget = max_source_length - len(question_ids) - 1

			if answer_start >= 0:
				answer_end = answer_start + len(example["target_texts"])
				answer_token_indices = [
					index for index, (start, end) in enumerate(offsets)
					if end > answer_start and start < answer_end
				]
				if not answer_token_indices:
					raise ValueError(
						"Could not align answer with context tokens for question: "
						f"{example['question_texts']}"
					)

				answer_token_start = answer_token_indices[0]
				answer_token_end = answer_token_indices[-1] + 1
				answer_token_count = answer_token_end - answer_token_start
				if answer_token_count > context_budget:
					raise ValueError(
						"Answer and question do not fit within max_source_length for "
						f"question: {example['question_texts']}"
					)

				# calulate context window indicies dynamically
				context_window_start = max(0, answer_token_start - (context_budget - answer_token_count) // 2)
				context_window_start = min(context_window_start, max(0, len(context_ids) - context_budget))

			context_window = context_ids[context_window_start:context_window_start + context_budget]
			input_ids = question_ids + context_window + [tokenizer.eos_token_id]
			features.append(
				{
					"input_ids": input_ids,
					"attention_mask": [1] * len(input_ids),
					"labels": labels,
				}
			)

		return collator(features)

	return collate


def train_qa(
	model: T5ForConditionalGeneration,
	train_dataloader: DataLoader,
	validation_dataloader: DataLoader,
	optimizer: torch.optim.Optimizer,
	epochs: int = 10,
):
	"""Fine-tune pretrained T5 with teacher forcing and its built-in loss"""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	for epoch in range(epochs):
		total_train_loss = 0.0
		train_batch_count = 0

		model.train()
		for batch in train_dataloader:
			batch = {name: value.to(device) for name, value in batch.items()}
			optimizer.zero_grad()
			loss = model(**batch).loss
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			total_train_loss += loss.item()
			train_batch_count += 1

		if train_batch_count == 0:
			raise RuntimeError(f"Training epoch {epoch + 1} produced no batches.")

		model.eval()
		total_validation_loss = 0.0
		validation_batch_count = 0
		with torch.no_grad():
			for batch in validation_dataloader:
				batch = {name: value.to(device) for name, value in batch.items()}
				total_validation_loss += model(**batch).loss.item()
				validation_batch_count += 1
		if validation_batch_count == 0:
			raise RuntimeError("Validation dataset produced no batches.")

		train_loss = total_train_loss / train_batch_count
		validation_loss = total_validation_loss / validation_batch_count
		print(
			f"epoch {epoch + 1:>3}/{epochs} "
			f"train_loss: {train_loss:.4f} "
			f"validation_loss: {validation_loss:.4f}"
		)

def make_prediction(
	model: T5ForConditionalGeneration,
	tokenizer: AutoTokenizer,
	question: str,
	context: str,
) -> str:
	
	"""Generate an answer from a question and context"""
	if not question.strip():
		raise ValueError("question must not be empty.")
	if not context.strip():
		raise ValueError("context must not be empty.")

	device = next(model.parameters()).device
	max_source_length = 512
	max_question_length = 128

	question_ids = tokenizer(
		f"question: {question} context:",
		add_special_tokens=False,
		truncation=True,
		max_length=max_question_length,
	)["input_ids"]

	context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
	context_budget = max_source_length - len(question_ids) - 1 # leave 1 space for <eos>
	if context_budget <= 0:
		raise ValueError("Question length is too long")

	# stride means 'overlap' between context, 128 by conventional
	stride = max(1, context_budget - 128)
	context_windows = []
	for start in range(0, len(context_ids), stride):
		window = context_ids[start:start + context_budget]
		context_windows.append(window)
		if start + context_budget >= len(context_ids):
			break

	input_features = [
		{
			"input_ids": question_ids + window + [tokenizer.eos_token_id],  # add <eos> at the end
			"attention_mask": [1] * (len(question_ids) + len(window) + 1),
		}
		for window in context_windows
	]
	model_inputs = tokenizer.pad(input_features, padding=True, return_tensors="pt")
	model_inputs = {name: value.to(device) for name, value in model_inputs.items()}

	model.eval()
	with torch.no_grad():
		generated = model.generate(
			**model_inputs,
			num_beams=4,
			max_new_tokens=64,
			return_dict_in_generate=True,
			output_scores=True,
		)

	predictions = tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)
	scores = generated.sequences_scores
	if scores is None:
		raise RuntimeError("Generation did not return sequence scores.")

	answer_candidates = [
		(prediction.strip(), score.item())
		for prediction, score in zip(predictions, scores)
		if prediction.strip().casefold() != "unanswerable"
	]
	if answer_candidates:
		return max(answer_candidates, key=lambda candidate: candidate[1])[0]
	return "unanswerable"

if __name__ == "__main__":
	torch.manual_seed(SEED)

	# using pretrained `t5-small` checkpoint for Q&A
	tokenizer_name = "google-t5/t5-small"
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
	model = T5ForConditionalGeneration.from_pretrained(tokenizer_name)
	source_dataset = load_dataset(
		"json",
		data_files=SQUAD_FILE,
		field="data",
		split="train",
	)

	# Split by article before flattening to keep related contexts in one split.
	source_splits = source_dataset.train_test_split(test_size=0.1, seed=SEED)
	train_dataset = TextToTextDataset(source_splits["train"])
	validation_dataset = TextToTextDataset(source_splits["test"])
	if len(train_dataset) == 0 or len(validation_dataset) == 0:
		raise ValueError("Need non-empty training and validation datasets.")
	optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

	shuffle_generator = torch.Generator().manual_seed(SEED)
	collate_fn = make_collate_fn(tokenizer, model)
	train_loader = DataLoader(
		train_dataset,
		batch_size=4,
		shuffle=True,
		generator=shuffle_generator,
		collate_fn=collate_fn,
	)
	validation_loader = DataLoader(
		validation_dataset,
		batch_size=4,
		shuffle=False,
		collate_fn=collate_fn,
	)
	train_qa(model, train_loader, validation_loader, optimizer, epochs=5)

	# Smoke-test generation with a real answerable example from the held-out split.
	question_texts = "Where is the world cup 2022 hosted?"
	context_texts = "The 2022 World Cup in Qatar is the 22nd World Cup. It is the first time in history that the World Cup has been held in Qatar and a country in the Middle East,and it is also the second in Asia."
	"In addition,the Qatar World Cup is the first time that the World Cup has been held in winter in the Northern Hemisphere and by a country that has never made it to the World Cup finals."
	prediction = make_prediction(
		model,
		tokenizer,
		question_texts,
		context_texts,
	)
	# print(f"Prediction test question: {sample['question_texts']}")
	print(f"Expected answer: Qatar")
	print(f"Predicted answer: {prediction}")
