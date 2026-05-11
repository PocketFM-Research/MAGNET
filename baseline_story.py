from __future__ import annotations

import argparse
import math
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
        "--timeout-seconds",
        type=int,
        default=int(os.getenv("BASELINE_STORY_TIMEOUT_SECONDS", "240")),
        help="HTTP timeout in seconds for the model request (default 240).",
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
        "Return only plain prose story text.\n\n"
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


def word_count(text: str) -> int:
    return len(text.split())


def _generate_single_part(
    llm: AnthropicLLM,
    *,
    system_prompt: str,
    base_user_prompt: str,
    pages: int,
    temperature: float = 0.35,
) -> dict[str, str]:
    prompt = base_user_prompt + "\n" + build_length_instruction(pages)
    story = _complete_plain_text(llm, system_prompt=system_prompt, user_prompt=prompt, temperature=temperature)
    if not story:
        raise RuntimeError("Model response missing non-empty `story`")
    return {
        "story": story,
    }


def _complete_plain_text(
    llm: AnthropicLLM,
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.35,
) -> str:
    url = f"{llm.base_url.rstrip('/')}/messages"
    omit_temperature = llm._should_omit_temperature()
    try:
        raw = llm._post_messages(url, system_prompt, user_prompt, temperature, omit_temperature)
    except Exception as exc:
        raise LLMError(f"Plain text completion failed: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError("Anthropic returned invalid JSON envelope") from exc
    content = llm._extract_text(parsed)
    if not content or not content.strip():
        raise LLMError("Anthropic response missing text content")
    llm._append_output_log(system_prompt, user_prompt, content)
    return content.strip()


def _generate_multi_part(
    llm: AnthropicLLM,
    *,
    system_prompt: str,
    base_user_prompt: str,
    total_pages: int,
) -> dict[str, object]:
    part_count = math.ceil(total_pages / 20)
    pages_per_part = [20] * part_count
    remainder = total_pages - (20 * (part_count - 1))
    pages_per_part[-1] = remainder

    parts: list[dict[str, object]] = []
    story_parts: list[str] = []
    carryover = ""
    canonical_title = "Baseline Story"

    for idx, pages in enumerate(pages_per_part, start=1):
        if idx == 1:
            part_instruction = (
                f"Write PART {idx}/{part_count} of one continuous novel-length story. "
                "This is the opening part, so establish setup naturally."
            )
        else:
            part_instruction = (
                f"Write PART {idx}/{part_count} of one continuous novel-length story. "
                "Continue directly from prior content with no recap headers."
            )
        if idx < part_count:
            part_instruction += " End this part with forward momentum into the next part."
        else:
            part_instruction += " This is the final part; deliver a satisfying ending."

        continuity_block = ""
        if carryover:
            continuity_block = (
                "\nCONTINUITY CONTEXT (continue from this exact trajectory):\n"
                f"{carryover}\n"
            )

        response = None
        attempt_pages = pages
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            text_guard = "Return only plain prose story text. No JSON, no markdown, no headers."
            part_prompt = (
                f"{base_user_prompt}\n\n{part_instruction}{continuity_block}\n"
                + build_length_instruction(attempt_pages)
                + "\n"
                + text_guard
            )
            try:
                part_story = _complete_plain_text(
                    llm,
                    system_prompt=system_prompt,
                    user_prompt=part_prompt,
                    temperature=0.35,
                )
                response = {"story": part_story}
                break
            except LLMError as exc:
                last_exc = exc
                # Most common failure here is truncated/invalid JSON on long generations.
                attempt_pages = max(5, attempt_pages // 2)
        if response is None:
            raise RuntimeError(f"Multipart generation failed at part {idx}: {last_exc}")

        part_story = str(response.get("story", "")).strip()
        if not part_story:
            raise RuntimeError(f"Model response missing non-empty `story` for part {idx}")

        story_parts.append(part_story)
        wc = word_count(part_story)
        parts.append({
            "part_index": idx,
            "target_pages": pages,
            "word_count_actual": wc,
        })
        carryover = part_story[-4000:]

    story = "\n\n".join(story_parts).strip()
    return {
        "title": canonical_title,
        "story": story,
        "parts": parts,
        "multipart": True,
        "part_count": part_count,
        "word_count_actual": word_count(story),
    }


def main() -> None:
    args = parse_args()
    if args.pages < 1:
        raise ValueError("--pages must be >= 1")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required")

    system_prompt, user_prompt = build_prompt_payload(args.fable_name)
    llm = AnthropicLLM(
        api_key=api_key,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout_seconds,
    )
    multipart = args.pages > 20
    try:
        if multipart:
            result = _generate_multi_part(
                llm,
                system_prompt=system_prompt,
                base_user_prompt=user_prompt,
                total_pages=args.pages,
            )
            title = str(result["title"])
            story = str(result["story"])
            parts = list(result["parts"])
            total_word_count_estimate = None
        else:
            single = _generate_single_part(
                llm,
                system_prompt=system_prompt,
                base_user_prompt=user_prompt,
                pages=args.pages,
            )
            title = "Baseline Story"
            story = str(single["story"])
            parts = []
            total_word_count_estimate = None
    except LLMError as exc:
        raise RuntimeError(f"Story generation failed: {exc}") from exc

    output_path = resolve_output_path(args.output, args.fable_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"[baseline-story] model={args.model}\n")
        handle.write(f"[fable] {args.fable_name}\n")
        handle.write(f"[title] {title}\n")
        handle.write(f"[target_pages] {args.pages}\n")
        handle.write(f"[multipart_generation] {str(multipart).lower()}\n")
        if multipart:
            handle.write(f"[parts] {len(parts)}\n")
            for part in parts:
                handle.write(
                    f"[part] index={part['part_index']} target_pages={part['target_pages']} "
                    f"word_count_actual={part['word_count_actual']}\n"
                )
        handle.write("=== STORY START ===\n")
        handle.write(story)
        handle.write("\n=== STORY END ===\n")

    actual_wc = word_count(story)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "model": args.model,
                "fable": args.fable_name,
                "target_pages": args.pages,
                "title": title,
                "multipart_generation": multipart,
                "parts": parts,
                "word_count_actual": actual_wc,
                "word_count_estimate": total_word_count_estimate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
