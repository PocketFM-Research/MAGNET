from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ACTION_SYSTEM_PREFIX = "You are the action generator for "
CRITIC_SYSTEM_PREFIX = "You are a strict action critic."


@dataclass
class ActionExample:
    source_path: str
    timestamp: str
    character: str
    system_prompt: str
    user_prompt: str
    action: str
    rationale: str
    confidence: float
    critic: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export action-generation SFT examples from llm_output logs."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=["llm_output.txt"],
        help="One or more llm_output log files to parse.",
    )
    parser.add_argument(
        "--output",
        default="data/action_sft_dataset.jsonl",
        help="Where to write the exported JSONL dataset.",
    )
    parser.add_argument(
        "--format",
        choices=["chat", "completion"],
        default="chat",
        help="Dataset shape to export.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Keep only action examples with model confidence >= this value.",
    )
    parser.add_argument(
        "--require-critic-approval",
        action="store_true",
        help="Keep only examples whose paired critic response has revise=false.",
    )
    parser.add_argument(
        "--require-goal-progress",
        action="store_true",
        help="Keep only examples whose paired critic response has advances_goal=true.",
    )
    parser.add_argument(
        "--dedupe-by-action",
        action="store_true",
        help="Drop exact duplicate normalized actions across all sources.",
    )
    parser.add_argument(
        "--include-rationale",
        action="store_true",
        help="Include the rationale field in the assistant target JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in args.input]
    examples: list[ActionExample] = []

    for path in input_paths:
        examples.extend(parse_log_file(path))

    filtered_examples = filter_examples(
        examples,
        min_confidence=args.min_confidence,
        require_critic_approval=args.require_critic_approval,
        require_goal_progress=args.require_goal_progress,
        dedupe_by_action=args.dedupe_by_action,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for example in filtered_examples:
            row = build_dataset_row(
                example=example,
                output_format=args.format,
                include_rationale=args.include_rationale,
            )
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(
        json.dumps(
            {
                "parsed_examples": len(examples),
                "written_examples": len(filtered_examples),
                "output_path": str(output_path),
                "format": args.format,
            },
            indent=2,
        )
    )


def parse_log_file(path: Path) -> list[ActionExample]:
    if not path.exists():
        raise FileNotFoundError(f"Input log file not found: {path}")

    text = path.read_text(encoding="utf-8")
    blocks = split_blocks(text)
    examples: list[ActionExample] = []
    pending_action: ActionExample | None = None

    for block in blocks:
        parsed = parse_block(block)
        if parsed is None:
            continue

        system_prompt = parsed["system_prompt"]
        if system_prompt.startswith(ACTION_SYSTEM_PREFIX):
            output = safe_json_loads(parsed["llm_output"])
            if not isinstance(output, dict):
                pending_action = None
                continue
            pending_action = ActionExample(
                source_path=str(path),
                timestamp=parsed["timestamp"],
                character=extract_field(parsed["user_prompt"], "Character") or "unknown",
                system_prompt=system_prompt,
                user_prompt=parsed["user_prompt"],
                action=str(output.get("action", "")).strip(),
                rationale=str(output.get("rationale", "")).strip(),
                confidence=coerce_float(output.get("confidence", 0.0)),
            )
            continue

        if system_prompt.startswith(CRITIC_SYSTEM_PREFIX) and pending_action is not None:
            critic_output = safe_json_loads(parsed["llm_output"])
            if isinstance(critic_output, dict):
                critic_character = extract_field(parsed["user_prompt"], "Character")
                if critic_character == pending_action.character:
                    pending_action.critic = critic_output
                    if pending_action.action:
                        examples.append(pending_action)
                    pending_action = None
            continue

    if pending_action is not None and pending_action.action:
        examples.append(pending_action)

    return examples


def split_blocks(text: str) -> list[str]:
    return [
        chunk.strip()
        for chunk in re.split(r"(?=^\[[^\n]+\] model=)", text, flags=re.MULTILINE)
        if chunk.strip()
    ]


def parse_block(block: str) -> dict[str, str] | None:
    pattern = re.compile(
        r"^\[(?P<timestamp>[^\]]+)\] model=(?P<model>[^\n]+)\n"
        r"=== SYSTEM PROMPT ===\n(?P<system_prompt>.*?)\n"
        r"=== USER PROMPT ===\n(?P<user_prompt>.*?)\n"
        r"=== LLM OUTPUT ===\n(?P<llm_output>.*?)\n=== END ===$",
        flags=re.DOTALL,
    )
    match = pattern.match(block)
    if not match:
        return None
    return match.groupdict()


def filter_examples(
    examples: list[ActionExample],
    min_confidence: float,
    require_critic_approval: bool,
    require_goal_progress: bool,
    dedupe_by_action: bool,
) -> list[ActionExample]:
    filtered: list[ActionExample] = []
    seen_actions: set[str] = set()

    for example in examples:
        if example.confidence < min_confidence:
            continue

        critic = example.critic or {}
        if require_critic_approval and bool(critic.get("revise", True)):
            continue
        if require_goal_progress and not bool(critic.get("advances_goal", False)):
            continue

        normalized_action = normalize_action(example.action)
        if dedupe_by_action:
            if normalized_action in seen_actions:
                continue
            seen_actions.add(normalized_action)

        filtered.append(example)

    return filtered


def build_dataset_row(
    example: ActionExample,
    output_format: str,
    include_rationale: bool,
) -> dict[str, Any]:
    completion_payload: dict[str, Any] = {"action": example.action}
    if include_rationale and example.rationale:
        completion_payload["rationale"] = example.rationale

    assistant_content = json.dumps(completion_payload, ensure_ascii=True)
    metadata = {
        "character": example.character,
        "source_path": example.source_path,
        "timestamp": example.timestamp,
        "confidence": example.confidence,
        "critic": example.critic or {},
    }

    if output_format == "completion":
        return {
            "prompt": build_serialized_prompt(example.system_prompt, example.user_prompt),
            "completion": assistant_content,
            "metadata": metadata,
        }

    return {
        "messages": [
            {"role": "system", "content": example.system_prompt},
            {"role": "user", "content": example.user_prompt},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": metadata,
    }


def build_serialized_prompt(system_prompt: str, user_prompt: str) -> str:
    return (
        "<start_of_turn>system\n"
        f"{system_prompt}\n"
        "<end_of_turn>\n"
        "<start_of_turn>user\n"
        f"{user_prompt}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def extract_field(user_prompt: str, field_name: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(field_name)}:\s*(.+)$", flags=re.MULTILINE)
    match = pattern.search(user_prompt)
    if not match:
        return None
    return match.group(1).strip()


def safe_json_loads(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def coerce_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def normalize_action(action: str) -> str:
    normalized = action.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


if __name__ == "__main__":
    main()

'''
Usage:

python scripts/export_action_sft_dataset.py --input llm_output.txt --output action_sft_dataset.jsonl --require-critic-approval --require-goal-progress --dedupe-by-action
'''