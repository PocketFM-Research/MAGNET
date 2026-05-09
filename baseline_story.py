from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fables import get_fable_definition
from llm import AnthropicLLM, LLMError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a baseline long-form story directly from a fable definition using Opus."
    )
    parser.add_argument(
        "--story",
        "--fable",
        dest="fable_name",
        default=os.getenv("FABLE_NAME", "missing_will"),
        help="Built-in story/fable name to use. Falls back to FABLE_NAME.",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=int(os.getenv("BASELINE_STORY_PAGES", "2")),
        help="Approximate target pages (default 2).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("BASELINE_STORY_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")),
        help="Anthropic model name (default BASELINE_STORY_MODEL/ANTHROPIC_MODEL/claude-opus-4-7).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=int(os.getenv("BASELINE_STORY_MAX_OUTPUT_TOKENS", "8192")),
        help="Max output tokens for Anthropic response (default 8192).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output txt path. Default: outputs/baseline_<fable>_<timestamp>.txt",
    )
    return parser.parse_args()


def build_prompt_payload(fable_name: str) -> tuple[str, str]:
    fable = get_fable_definition(fable_name)
    character_payload = [asdict(character) for character in fable.characters]

    # Rough prose page estimate: ~500 words/page.
    system = (
        "You are a strong long-form fiction writer. "
        "Use only the provided fable context. "
        "Return only JSON."
    )
    user = (
        "Write one cohesive story based on the provided fable definition. "
        "Do not include outlines, section labels, or analysis. "
        "Keep character behavior consistent with their profile and world state.\n\n"
        "Return JSON keys:\n"
        "- title (string)\n"
        "- story (string, plain prose)\n"
        "- word_count_estimate (integer)\n\n"
        f"FABLE_NAME: {fable.name}\n"
        f"STORY_GOAL: {fable.goal}\n"
        f"INITIAL_WORLD_VARS: {json.dumps(fable.initial_world_vars, ensure_ascii=True, sort_keys=True)}\n"
        f"CHARACTERS: {json.dumps(character_payload, ensure_ascii=True, sort_keys=True)}\n"
    )
    return system, user


def build_length_instruction(pages: int) -> str:
    pages = max(1, pages)
    target_words = pages * 500
    low = max(300, int(target_words * 0.85))
    high = int(target_words * 1.15)
    return (
        f"Length target: about {pages} page(s), roughly {low}-{high} words. "
        "Prioritize complete dramatic arc and coherence over exact count."
    )


def resolve_output_path(path_arg: str | None, fable_name: str) -> Path:
    if path_arg:
        return Path(path_arg)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("outputs") / f"baseline_{fable_name}_{ts}.txt"


def main() -> None:
    args = parse_args()
    if args.pages < 1:
        raise ValueError("--pages must be >= 1")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required")

    system_prompt, user_prompt = build_prompt_payload(args.fable_name)
    user_prompt = user_prompt + "\n" + build_length_instruction(args.pages)

    llm = AnthropicLLM(
        api_key=api_key,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
    )
    try:
        response = llm.complete_json(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.35)
    except LLMError as exc:
        raise RuntimeError(f"Story generation failed: {exc}") from exc

    title = str(response.get("title", "Untitled Baseline Story")).strip() or "Untitled Baseline Story"
    story = str(response.get("story", "")).strip()
    if not story:
        raise RuntimeError("Model response missing non-empty `story`")

    output_path = resolve_output_path(args.output, args.fable_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[baseline-story] model={args.model}\n")
        handle.write(f"[fable] {args.fable_name}\n")
        handle.write(f"[title] {title}\n")
        handle.write(f"[target_pages] {args.pages}\n")
        handle.write("=== STORY START ===\n")
        handle.write(story)
        handle.write("\n=== STORY END ===\n")

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "model": args.model,
                "fable": args.fable_name,
                "target_pages": args.pages,
                "title": title,
                "word_count_estimate": response.get("word_count_estimate"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
