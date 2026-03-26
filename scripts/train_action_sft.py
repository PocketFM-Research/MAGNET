from __future__ import annotations

import argparse
import json
import inspect
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


DEFAULT_MODEL = "google/gemma-2-2b-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA/QLoRA SFT for character action generation."
    )
    parser.add_argument(
        "--dataset-path",
        default="data/action_sft_dataset.jsonl",
        help="Path to the exported JSONL dataset.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="Base Gemma checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/gemma-action-sft",
        help="Directory for checkpoints and final adapter weights.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1024,
        help="Maximum tokenized sequence length.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate for LoRA training.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=3.0,
        help="Number of train epochs.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="How often to log train metrics.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="How often to save checkpoints.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Load the base model in 4-bit for QLoRA training.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of examples to reserve for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run scripts/export_action_sft_dataset.py first."
        )

    device_backend = detect_device_backend()
    if args.use_4bit and device_backend != "cuda":
        raise ValueError(
            "--use-4bit is only supported on NVIDIA CUDA GPUs in this script. "
            f"Detected backend: {device_backend}. On Apple Silicon, run without --use-4bit."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {}
    if device_backend == "cuda":
        model_kwargs["device_map"] = "auto"
        model_kwargs["dtype"] = torch.bfloat16
    elif device_backend == "mps":
        model_kwargs["dtype"] = torch.float16
    else:
        model_kwargs["dtype"] = torch.float32

    if args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    elif device_backend == "mps":
        model = model.to("mps")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    split_dataset = dataset.train_test_split(
        test_size=args.validation_split,
        seed=args.seed,
        shuffle=True,
    )

    train_dataset = prepare_tokenized_dataset(
        split_dataset["train"], tokenizer, args.max_seq_length
    )
    eval_dataset = prepare_tokenized_dataset(
        split_dataset["test"], tokenizer, args.max_seq_length
    )

    training_kwargs = dict(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        save_total_limit=2,
        bf16=device_backend == "cuda",
        fp16=device_backend == "mps",
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )
    if device_backend == "cpu":
        training_kwargs["use_cpu"] = True
    if device_backend == "mps":
        training_kwargs["use_mps_device"] = True
    strategy_arg = resolve_eval_strategy_arg()
    training_kwargs[strategy_arg] = "steps"
    training_args = TrainingArguments(**filter_training_kwargs(training_kwargs))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
        ),
    )

    model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    summary_path = Path(args.output_dir) / "training_run_config.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
        handle.write("\n")

    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "train_examples": len(train_dataset),
                "eval_examples": len(eval_dataset),
                "model_name": args.model_name,
                "use_4bit": args.use_4bit,
                "device_backend": device_backend,
            },
            indent=2,
        )
    )


def prepare_tokenized_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> Dataset:
    formatted = dataset.map(
        lambda row: build_training_example(row["messages"], tokenizer, max_seq_length),
        remove_columns=dataset.column_names,
    )
    return formatted


def detect_device_backend() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_eval_strategy_arg() -> str:
    signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in signature.parameters:
        return "evaluation_strategy"
    if "eval_strategy" in signature.parameters:
        return "eval_strategy"
    raise TypeError(
        "TrainingArguments does not support either `evaluation_strategy` or `eval_strategy`."
    )


def filter_training_kwargs(training_kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(TrainingArguments.__init__)
    supported = set(signature.parameters.keys())
    return {key: value for key, value in training_kwargs.items() if key in supported}


def build_training_example(
    messages: list[dict[str, str]],
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> dict[str, Any]:
    normalized_messages = normalize_messages_for_template(messages)
    prompt_messages = normalized_messages[:-1]
    full_text = tokenizer.apply_chat_template(
        normalized_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )
    prompt_tokens = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )

    prompt_length = min(len(prompt_tokens["input_ids"]), len(full_tokens["input_ids"]))
    labels = list(full_tokens["input_ids"])
    for idx in range(prompt_length):
        labels[idx] = -100

    full_tokens["labels"] = labels
    return full_tokens


def normalize_messages_for_template(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if not messages:
        return messages

    normalized: list[dict[str, str]] = []
    pending_system_parts: list[str] = []

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "system":
            pending_system_parts.append(content.strip())
            continue

        if role == "user" and pending_system_parts:
            system_prefix = "\n\n".join(
                f"System instructions:\n{part}" for part in pending_system_parts if part
            ).strip()
            merged_content = f"{system_prefix}\n\n{content}".strip()
            normalized.append({"role": "user", "content": merged_content})
            pending_system_parts = []
            continue

        normalized.append({"role": role, "content": content})

    if pending_system_parts:
        normalized.insert(
            0,
            {
                "role": "user",
                "content": "\n\n".join(
                    f"System instructions:\n{part}" for part in pending_system_parts if part
                ).strip(),
            },
        )

    return normalized


if __name__ == "__main__":
    main()
