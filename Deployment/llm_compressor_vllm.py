"""Quantize a small instruction model with LLM Compressor and run it with vLLM."""

import argparse
import gc

import torch
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
CALIBRATION_DATASET = "HuggingFaceH4/ultrachat_200k"
NUM_CALIBRATION_SAMPLES = 8
MAX_SEQUENCE_LENGTH = 512


def main():
    parser = argparse.ArgumentParser(
        description="Quantize a model with LLM Compressor and serve it with vLLM."
    )
    parser.add_argument(
        "--output-dir",
        default="compressed_model",
        help="Directory where the compressed model will be saved.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    calibration_data = load_dataset(
        CALIBRATION_DATASET,
        split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]",
    ).shuffle(seed=42)

    def format_chat(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False
            )
        }

    calibration_data = calibration_data.map(format_chat)

    def tokenize(example):
        return tokenizer(
            example["text"],
            padding=False,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )

    calibration_data = calibration_data.map(
        tokenize, remove_columns=calibration_data.column_names
    )

    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head"],
    )
    oneshot(
        model=model,
        dataset=calibration_data,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    model.save_pretrained(args.output_dir, save_compressed=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Compressed model saved to {args.output_dir}")

    del model, tokenizer, calibration_data
    gc.collect()
    torch.cuda.empty_cache()

    engine = LLM(
        model=args.output_dir,
        max_model_len=MAX_SEQUENCE_LENGTH,
        quantization="compressed-tensors",
    )
    results = engine.generate(
        ["Explain in one sentence why quantization can reduce model memory usage."],
        SamplingParams(temperature=0.0, max_tokens=64),
    )
    print(f"Prompt: {results[0].prompt}")
    print(f"Response: {results[0].outputs[0].text.strip()}")


if __name__ == "__main__":
    main()
