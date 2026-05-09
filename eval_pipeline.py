from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


CATEGORY_ORDER = [
    "story structure",
    "character actions and emotions",
    "sentence writing",
    "logic and consistency",
]


class EvalError(RuntimeError):
    pass


@dataclass
class OpenAILLM:
    api_key: str
    model: str = "gpt-5.4-mini"
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: int = 60

    def complete_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise EvalError(f"OpenAI HTTP {exc.code}: {raw[:500]}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EvalError("OpenAI returned non-JSON response envelope") from exc

        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise EvalError(f"OpenAI response missing choices: {raw[:500]}")
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise EvalError(f"OpenAI response missing content: {raw[:500]}")

        return _parse_json_object(content)


def _parse_json_object(content: str) -> dict[str, Any]:
    candidates = [content.strip()]
    extracted = _extract_first_json_object(content)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    for candidate in candidates:
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise EvalError("Model output was not a valid JSON object")


def _extract_first_json_object(content: str) -> str | None:
    start = content.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(content)):
        ch = content[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return content[start : idx + 1]

    return None


def extract_story_block(text: str) -> str:
    pattern = re.compile(r"=== STORY START ===\s*(.*?)\s*=== STORY END ===", flags=re.DOTALL)
    matches = list(pattern.finditer(text))
    if not matches:
        raise EvalError("Could not find story block between STORY START/END markers")

    story = matches[-1].group(1).strip()
    if not story:
        raise EvalError("Story block is present but empty")
    return story


def _editor_prompts(story: str) -> tuple[str, str]:
    categories = ", ".join(CATEGORY_ORDER)
    system = (
        "You are an expert story editor. "
        "Return only JSON. "
        "Each comment must be a specific, actionable critique tied to this story text. "
        f"Allowed categories: {categories}."
    )
    user = (
        "Read the story and annotate editor comments. "
        "Return JSON with key `comments`, where `comments` is an array of objects with keys: "
        "`category` (one allowed category), `comment` (string), `evidence` (short quote or reference). "
        # "Include up to 1 annotation per paragraph. \n\n"
        "STORY:\n"
        f"{story}"
    )
    return system, user


def normalize_comments(payload: dict[str, Any]) -> list[dict[str, str]]:
    raw_comments = payload.get("comments")
    if not isinstance(raw_comments, list):
        raise EvalError("Model JSON must include a `comments` array")

    normalized: list[dict[str, str]] = []
    allowed = set(CATEGORY_ORDER)
    for item in raw_comments:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category", "")).strip().lower()
        comment = str(item.get("comment", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        if category not in allowed or not comment:
            continue
        normalized.append({
            "category": category,
            "comment": comment,
            "evidence": evidence,
        })
    return normalized


def count_by_category(comments: list[dict[str, str]]) -> dict[str, int]:
    counts = {key: 0 for key in CATEGORY_ORDER}
    for comment in comments:
        category = comment["category"]
        if category in counts:
            counts[category] += 1
    return counts


def evaluate_story_file(path: Path, llm: OpenAILLM) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    story = extract_story_block(text)
    system, user = _editor_prompts(story)
    payload = llm.complete_json(system_prompt=system, user_prompt=user, temperature=0.1)
    comments = normalize_comments(payload)
    counts = count_by_category(comments)
    return {
        "file": str(path),
        "story": story,
        "comments": comments,
        "counts": counts,
        "total_comments": len(comments),
    }


def build_comparison(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    deltas = {
        cat: int(b["counts"].get(cat, 0)) - int(a["counts"].get(cat, 0))
        for cat in CATEGORY_ORDER
    }
    return {
        "baseline": a["file"],
        "candidate": b["file"],
        "baseline_counts": a["counts"],
        "candidate_counts": b["counts"],
        "delta_candidate_minus_baseline": deltas,
        "baseline_total_comments": a["total_comments"],
        "candidate_total_comments": b["total_comments"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate story quality comments from one or two txt files. "
            "The story is extracted from the last STORY START/END block."
        )
    )
    parser.add_argument("one", help="Path to first txt file")
    parser.add_argument("two", nargs="?", help="Optional second txt file to compare")
    parser.add_argument(
        "--model",
        default=os.getenv("EVAL_LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.4-mini")),
        help="OpenAI model name (default: EVAL_LLM_MODEL/OPENAI_MODEL/gpt-5.4-mini)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible API base URL (default: OPENAI_BASE_URL or https://api.openai.com/v1)",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("EVAL_OUTPUT_PATH", "eval_output.txt"),
        help="Path to write evaluation JSON output (default: EVAL_OUTPUT_PATH or eval_output.txt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EvalError("OPENAI_API_KEY is required")

    llm = OpenAILLM(api_key=api_key, model=args.model, base_url=args.base_url)

    first = evaluate_story_file(Path(args.one), llm)
    output: dict[str, Any] = {"one": first}

    if args.two:
        second = evaluate_story_file(Path(args.two), llm)
        output["two"] = second
        output["comparison"] = build_comparison(first, second)

    rendered = json.dumps(output, indent=2, ensure_ascii=False)
    print(rendered)
    Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
