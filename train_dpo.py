from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from prompts import build_action_prompt


class TextOnlyProcessingAdapter:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.pop("images", None)
        text = kwargs.pop("text", None)
        if text is not None and not args:
            return self.tokenizer(text, *args, **kwargs)
        return self.tokenizer(*args, **kwargs)

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DPO adapter from generated world-model preference data."
    )
    parser.add_argument(
        "--dataset",
        default="artifacts/dpo_preferences.jsonl",
        help="Path to the JSONL preference dataset.",
    )
    parser.add_argument(
        "--model",
        default="google/gemma-2-2b-it",
        help="Base model to fine-tune with DPO.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/gemma-action-dpo",
        help="Directory where the trained adapter will be saved.",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Model context length.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device train batch size.")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta.")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Warmup steps.")
    parser.add_argument("--logging-steps", type=int, default=5, help="Trainer logging frequency.")
    parser.add_argument("--save-steps", type=int, default=50, help="Checkpoint save frequency.")
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.05,
        help="Fraction of rows reserved for evaluation. Use 0 to disable eval.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling and splits.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of rows to load. 0 keeps the full dataset.",
    )
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument(
        "--default-confidence",
        type=float,
        default=0.7,
        help="Confidence value injected into chosen/rejected JSON completions.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the base model in 4-bit mode via bitsandbytes.",
    )
    parser.add_argument(
        "--lora-target-modules",
        default="auto",
        help=(
            "LoRA target modules: 'auto', 'attention-only', or a comma-separated list "
            "such as q_proj,k_proj,v_proj,o_proj."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    dataset_path = Path(args.dataset)
    rows = load_rows(dataset_path, limit=args.limit)
    if not rows:
        raise RuntimeError(f"No training rows found in {dataset_path}")

    model_kwargs: dict[str, Any] = {}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer = ensure_trl_tokenizer_compat(tokenizer)
    processing_class = build_trl_processing_class(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = resolve_lora_target_modules(model, args.lora_target_modules)
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    ensure_trl_model_compat(model)

    converted = [
        convert_row_to_preference_example(
            row,
            tokenizer=tokenizer,
            default_confidence=args.default_confidence,
        )
        for row in rows
    ]
    train_rows, eval_rows = split_rows(converted, eval_fraction=args.eval_fraction, seed=args.seed)
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "beta": args.beta,
        "warmup_steps": args.warmup_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_strategy": "steps" if eval_dataset is not None else "no",
        "save_total_limit": 2,
        "lr_scheduler_type": "cosine",
        "report_to": "none",
        "seed": args.seed,
        "max_prompt_length": args.max_seq_length // 2,
        "max_length": args.max_seq_length,
        "remove_unused_columns": False,
    }
    if eval_dataset is not None:
        training_kwargs["eval_steps"] = args.save_steps
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        training_kwargs["bf16"] = True
    else:
        training_kwargs["fp16"] = torch.cuda.is_available()

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=DPOConfig(**training_kwargs),
        processing_class=processing_class,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
                "output_dir": str(output_dir),
                "model": args.model,
                "lora_target_modules": target_modules,
            },
            indent=2,
        )
    )


def load_rows(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def convert_row_to_preference_example(
    row: dict[str, Any],
    tokenizer: Any,
    default_confidence: float,
) -> dict[str, str]:
    payload = row.get("prompt", {})
    if not isinstance(payload, dict):
        raise ValueError("Each dataset row must contain a prompt object.")

    system_prompt, user_prompt = build_action_prompt(
        str(payload.get("character", "")),
        str(payload.get("persona", "")),
        str(payload.get("goal", "")),
        payload.get("world_vars", {}) if isinstance(payload.get("world_vars", {}), dict) else {},
        list(payload.get("memory_snippets", [])) if isinstance(payload.get("memory_snippets", []), list) else [],
        list(payload.get("world_knowledge", [])) if isinstance(payload.get("world_knowledge", []), list) else [],
        list(payload.get("recent_story", [])) if isinstance(payload.get("recent_story", []), list) else [],
        None,
    )

    prompt = build_chat_prompt(tokenizer, system_prompt, user_prompt)
    chosen = build_completion_text(
        action=str(row.get("chosen", "")).strip(),
        rationale=str(row.get("chosen_rationale", "")).strip(),
        confidence=default_confidence,
    )
    rejected = build_completion_text(
        action=str(row.get("rejected", "")).strip(),
        rationale=str(row.get("rejected_rationale", "")).strip(),
        confidence=max(0.0, min(1.0, default_confidence - 0.2)),
    )
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def build_chat_prompt(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return (
            "<system>\n"
            f"{system_prompt}\n"
            "</system>\n"
            "<user>\n"
            f"{user_prompt}\n"
            "</user>\n"
            "<assistant>\n"
        )


def build_completion_text(action: str, rationale: str, confidence: float) -> str:
    payload = {
        "action": action or "look around",
        "confidence": round(max(0.0, min(1.0, confidence)), 2),
        "rationale": rationale or "fallback",
    }
    return json.dumps(payload, ensure_ascii=True)


def ensure_trl_model_compat(model: Any) -> None:
    if not hasattr(model, "warnings_issued") or not isinstance(getattr(model, "warnings_issued"), dict):
        setattr(model, "warnings_issued", {})


def ensure_trl_tokenizer_compat(tokenizer: Any) -> Any:
    if not hasattr(tokenizer, "tokenizer"):
        setattr(tokenizer, "tokenizer", tokenizer)
    return tokenizer


def build_trl_processing_class(tokenizer: Any) -> Any:
    return TextOnlyProcessingAdapter(tokenizer)


def resolve_lora_target_modules(model: Any, requested: str) -> list[str]:
    normalized = requested.strip().lower()
    if not normalized or normalized == "auto":
        return discover_lora_target_modules(model, include_mlp=True)
    if normalized == "attention-only":
        return discover_lora_target_modules(model, include_mlp=False)

    modules = [part.strip() for part in requested.split(",") if part.strip()]
    if not modules:
        raise ValueError("No valid LoRA target modules were provided.")
    return expand_requested_target_modules(model, modules)


def discover_lora_target_modules(model: Any, include_mlp: bool) -> list[str]:
    attention_suffixes = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "qkv_proj",
        "query_key_value",
        "c_attn",
        "c_proj",
        "Wqkv",
    ]
    mlp_suffixes = [
        "gate_proj",
        "up_proj",
        "down_proj",
        "gate_up_proj",
        "wi",
        "wo",
        "w1",
        "w2",
        "w3",
        "ffn_up_proj",
        "ffn_down_proj",
    ]
    candidate_suffixes = attention_suffixes + (mlp_suffixes if include_mlp else [])
    excluded_suffixes = {"lm_head", "embed_tokens", "tok_embeddings", "output"}

    discovered: list[str] = []
    discovered_wrapped: list[str] = []
    for name, module in model.named_modules():
        if not name or "." not in name:
            continue
        suffix = name.rsplit(".", 1)[-1]
        parent_name = name.rsplit(".", 1)[0]
        parent_suffix = parent_name.rsplit(".", 1)[-1] if "." in parent_name else parent_name
        if suffix in excluded_suffixes:
            continue
        if suffix == "linear" and parent_suffix in candidate_suffixes and hasattr(module, "weight"):
            if name not in discovered_wrapped:
                discovered_wrapped.append(name)
            continue
        if suffix not in candidate_suffixes:
            continue
        if not hasattr(module, "weight"):
            continue
        if suffix not in discovered:
            discovered.append(suffix)

    if discovered_wrapped:
        return discovered_wrapped

    if discovered:
        return discovered

    fallback = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if include_mlp:
        fallback.extend(["gate_proj", "up_proj", "down_proj"])
    return fallback


def expand_requested_target_modules(model: Any, modules: list[str]) -> list[str]:
    requested = set(modules)
    expanded: list[str] = []
    for name, module in model.named_modules():
        if not name or "." not in name:
            continue
        suffix = name.rsplit(".", 1)[-1]
        parent_name = name.rsplit(".", 1)[0]
        parent_suffix = parent_name.rsplit(".", 1)[-1] if "." in parent_name else parent_name
        if suffix == "linear" and parent_suffix in requested and hasattr(module, "weight"):
            if name not in expanded:
                expanded.append(name)

    if expanded:
        return expanded
    return modules


def split_rows(
    rows: list[dict[str, str]],
    eval_fraction: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    if eval_fraction <= 0 or len(shuffled) < 20:
        return shuffled, []

    eval_size = max(1, int(len(shuffled) * eval_fraction))
    if eval_size >= len(shuffled):
        eval_size = max(1, len(shuffled) // 10)

    eval_rows = shuffled[:eval_size]
    train_rows = shuffled[eval_size:]
    return train_rows, eval_rows


if __name__ == "__main__":
    main()
