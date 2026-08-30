from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from llm import AnthropicLLM, GeminiLLM, LLMError


CATEGORY_ORDER = [
    "story",
    "chapter",
    "sentence",
]

LEVEL_CATEGORIES: dict[str, list[str]] = {
    "story": [
        "logical consistency",
        "thematic coherence",
        "character arc completion",
    ],
    "chapter": [
        "goal conflict outcome",
        "hook and close",
        "chapter necessity",
    ],
    "sentence": [
        "rhythm",
        "clarity",
        "syntax variety",
    ],
}

CHAPTER_WORD_TARGET = 2000
CHAPTER_SAMPLE_COUNT = 5
SENTENCE_SAMPLE_COUNT = 5
MAX_COMMENTS_PER_EVAL = 40
LLM_RETRY_ATTEMPTS = 3

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


EvalLLM = OpenAILLM | GeminiLLM | AnthropicLLM

DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"


def _normalize_provider_name(raw_provider: str | None) -> str:
    provider = (raw_provider or "").strip().lower()
    if provider in {"", "openai"}:
        return "openai"
    if provider in {"gemini", "google"}:
        return "gemini"
    if provider in {"anthropic", "claude"}:
        return "anthropic"
    raise EvalError(f"Unsupported LLM provider: {raw_provider}")


def _default_model_for_provider(provider: str) -> str:
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
    return os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)


def build_eval_llm(
    provider: str,
    model: str | None = None,
    base_url: str | None = None,
    timeout_seconds: int = 600,
) -> EvalLLM:
    normalized_provider = _normalize_provider_name(provider)
    resolved_model = model or _default_model_for_provider(normalized_provider)

    if normalized_provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EvalError("GEMINI_API_KEY is required for Gemini provider")
        return GeminiLLM(
            api_key=api_key,
            model=resolved_model,
            base_url=base_url or os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
            timeout_seconds=timeout_seconds,
            output_log_path=None,
        )

    if normalized_provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EvalError("ANTHROPIC_API_KEY is required for Anthropic provider")
        return AnthropicLLM(
            api_key=api_key,
            model=resolved_model,
            base_url=base_url or os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
            timeout_seconds=timeout_seconds,
            output_log_path=None,
            max_output_tokens=int(os.getenv("ANTHROPIC_MAX_OUTPUT_TOKENS", "8192")),
            anthropic_version=os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EvalError("OPENAI_API_KEY is required for OpenAI provider")
    return OpenAILLM(
        api_key=api_key,
        model=resolved_model,
        base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        timeout_seconds=timeout_seconds,
    )


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


def chunk_story_by_words(story: str, target_words: int = CHAPTER_WORD_TARGET) -> list[dict[str, Any]]:
    raw_paras = [p.strip() for p in re.split(r"\n\s*\n", story) if p.strip()]
    paras: list[str] = []
    for para in raw_paras:
        para_words_list = para.split()
        if len(para_words_list) <= target_words:
            paras.append(para)
            continue

        sentence_parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
        if len(sentence_parts) <= 1:
            for i in range(0, len(para_words_list), target_words):
                paras.append(" ".join(para_words_list[i : i + target_words]).strip())
            continue

        current_sentences: list[str] = []
        current_words = 0
        for sentence in sentence_parts:
            sentence_words = len(sentence.split())
            if current_sentences and current_words + sentence_words > target_words:
                paras.append(" ".join(current_sentences).strip())
                current_sentences = [sentence]
                current_words = sentence_words
            else:
                current_sentences.append(sentence)
                current_words += sentence_words
        if current_sentences:
            paras.append(" ".join(current_sentences).strip())

    chunks: list[dict[str, Any]] = []
    current_paras: list[str] = []
    current_words = 0
    start_word = 1

    for para in paras:
        para_words = len(para.split())
        if current_paras and current_words + para_words > target_words:
            text = "\n\n".join(current_paras).strip()
            end_word = start_word + current_words - 1
            chunks.append({
                "index": len(chunks) + 1,
                "start_word": start_word,
                "end_word": end_word,
                "word_count": current_words,
                "text": text,
            })
            start_word = end_word + 1
            current_paras = [para]
            current_words = para_words
        else:
            current_paras.append(para)
            current_words += para_words

    if current_paras:
        text = "\n\n".join(current_paras).strip()
        end_word = start_word + current_words - 1
        chunks.append({
            "index": len(chunks) + 1,
            "start_word": start_word,
            "end_word": end_word,
            "word_count": current_words,
            "text": text,
        })

    return chunks


def split_sentences(story: str) -> list[str]:
    story = re.sub(r"\s+", " ", story).strip()
    if not story:
        return []
    parts = re.split(r"(?<=[.!?])\s+", story)
    return [p.strip() for p in parts if p.strip()]


def sample_sentences(sentences: list[str], sample_count: int = SENTENCE_SAMPLE_COUNT) -> list[dict[str, Any]]:
    if not sentences:
        return []
    if len(sentences) <= sample_count:
        selected = list(range(len(sentences)))
    else:
        selected = sorted({round(i * (len(sentences) - 1) / (sample_count - 1)) for i in range(sample_count)})

    out: list[dict[str, Any]] = []
    for idx in selected:
        out.append({
            "index": idx + 1,
            "text": sentences[idx],
        })
    return out


def sample_chapters(
    chapters: list[dict[str, Any]],
    sample_count: int = CHAPTER_SAMPLE_COUNT,
    *,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not chapters:
        return []
    if len(chapters) <= sample_count:
        selected = list(range(len(chapters)))
        selected.extend(rng.randrange(len(chapters)) for _ in range(sample_count - len(chapters)))
    else:
        selected = sorted({round(i * (len(chapters) - 1) / (sample_count - 1)) for i in range(sample_count)})
    return [chapters[idx] for idx in selected]


def _editor_prompts(level: str, content_text: str, context: str = "") -> tuple[str, str]:
    categories = ", ".join(LEVEL_CATEGORIES[level])
    system = (
        "You are an expert story editor. "
        "Return only a single JSON object. "
        "Do not write any introduction, praise, summary, explanation, markdown fence, or trailing note. "
        "If you are unsure, return an empty `comments` array rather than prose. "
        "Each comment must be a specific, actionable critique tied to the provided text. "
        f"Allowed categories: {categories}."
    )
    context_block = f"CONTEXT:\n{context}\n\n" if context else ""
    user = (
        f"Read the {level}-level text and annotate editor comments. "
        f"Include up to {MAX_COMMENTS_PER_EVAL} comments. "
        "Return JSON with key `comments`, where `comments` is an array of objects with keys: "
        "`category` (one allowed category), `comment` (string), `evidence` (short quote or reference). "
        "If there are no strong comments, return exactly `{\"comments\":[]}`. "
        f"{context_block}"
        f"TEXT:\n{content_text}"
    )
    return system, user


def _complete_json_with_retries(
    llm: EvalLLM,
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    attempts: int = LLM_RETRY_ATTEMPTS,
) -> dict[str, Any]:
    last_error: Exception | None = None
    retry_system_prompt = system_prompt
    retry_user_prompt = user_prompt
    for attempt in range(1, attempts + 1):
        try:
            return llm.complete_json(
                system_prompt=retry_system_prompt,
                user_prompt=retry_user_prompt,
                temperature=temperature,
            )
        except (EvalError, LLMError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            retry_system_prompt = (
                f"{system_prompt}\n"
                "Your previous attempt was invalid because it was not parseable as a single JSON object. "
                "Retry now and return JSON only."
            )
            retry_user_prompt = (
                f"{user_prompt}\n\n"
                "IMPORTANT RETRY INSTRUCTION: Return exactly one JSON object and nothing else. "
                "Do not include commentary before or after the JSON."
            )
    if last_error is None:
        raise EvalError("LLM request failed without an error")
    raise EvalError(f"LLM request failed after {attempts} attempts: {last_error}") from last_error


def normalize_comments(payload: dict[str, Any], level: str) -> list[dict[str, str]]:
    raw_comments = payload.get("comments")
    if not isinstance(raw_comments, list):
        raise EvalError("Model JSON must include a `comments` array")

    normalized: list[dict[str, str]] = []
    allowed = set(LEVEL_CATEGORIES[level])
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


def count_by_category(comments: list[dict[str, str]], categories: list[str]) -> dict[str, int]:
    counts = {key: 0 for key in categories}
    for comment in comments:
        category = comment["category"]
        if category in counts:
            counts[category] += 1
    return counts


def evaluate_text_block(
    llm: EvalLLM,
    *,
    level: str,
    content_text: str,
    context: str = "",
    source_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    system, user = _editor_prompts(level=level, content_text=content_text, context=context)
    payload = _complete_json_with_retries(llm, system_prompt=system, user_prompt=user, temperature=0.1)
    comments = normalize_comments(payload, level=level)
    counts = count_by_category(comments, LEVEL_CATEGORIES[level])
    return {
        "level": level,
        "source_meta": source_meta or {},
        "comments": comments,
        "counts": counts,
        "total_comments": len(comments),
    }


def evaluate_story_file(path: Path, llm: EvalLLM, *, rng: random.Random) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    story = extract_story_block(text)
    story_eval = evaluate_text_block(
        llm,
        level="story",
        content_text=story,
        source_meta={"unit": "full_story"},
    )

    chapters = chunk_story_by_words(story, target_words=CHAPTER_WORD_TARGET)
    sampled_chapters = sample_chapters(chapters, sample_count=CHAPTER_SAMPLE_COUNT, rng=rng)
    chapter_evals: list[dict[str, Any]] = []
    for chapter in sampled_chapters:
        chapter_eval = evaluate_text_block(
            llm,
            level="chapter",
            content_text=chapter["text"],
            context=f"chapter_index={chapter['index']}, word_span={chapter['start_word']}-{chapter['end_word']}",
            source_meta={
                "chapter_index": chapter["index"],
                "start_word": chapter["start_word"],
                "end_word": chapter["end_word"],
                "word_count": chapter["word_count"],
            },
        )
        chapter_evals.append(chapter_eval)

    all_sentences = split_sentences(story)
    sentence_samples = sample_sentences(all_sentences, sample_count=SENTENCE_SAMPLE_COUNT)
    sentence_evals: list[dict[str, Any]] = []
    for sample in sentence_samples:
        sentence_eval = evaluate_text_block(
            llm,
            level="sentence",
            content_text=sample["text"],
            context=f"sentence_index={sample['index']}",
            source_meta={"sentence_index": sample["index"]},
        )
        sentence_evals.append(sentence_eval)

    chapter_counts = {cat: 0 for cat in LEVEL_CATEGORIES["chapter"]}
    for item in chapter_evals:
        for cat, value in item["counts"].items():
            chapter_counts[cat] += int(value)

    sentence_counts = {cat: 0 for cat in LEVEL_CATEGORIES["sentence"]}
    for item in sentence_evals:
        for cat, value in item["counts"].items():
            sentence_counts[cat] += int(value)

    counts = {
        "story": story_eval["counts"],
        "chapter": chapter_counts,
        "sentence": sentence_counts,
    }
    total_comments = (
        int(story_eval["total_comments"])
        + sum(int(item["total_comments"]) for item in chapter_evals)
        + sum(int(item["total_comments"]) for item in sentence_evals)
    )

    return {
        "file": str(path),
        "story": story,
        "level_categories": LEVEL_CATEGORIES,
        "story_eval": story_eval,
        "chapter_plan": {
            "target_words": CHAPTER_WORD_TARGET,
            "sample_count_target": CHAPTER_SAMPLE_COUNT,
            "num_chapters_total": len(chapters),
            "num_chapters_sampled": len(sampled_chapters),
            "sampled_chapter_indices": [c["index"] for c in sampled_chapters],
            "chapters": [
                {
                    "chapter_index": c["index"],
                    "start_word": c["start_word"],
                    "end_word": c["end_word"],
                    "word_count": c["word_count"],
                }
                for c in chapters
            ],
        },
        "chapter_evals": chapter_evals,
        "sentence_plan": {
            "sample_count_target": SENTENCE_SAMPLE_COUNT,
            "num_story_sentences": len(all_sentences),
            "num_sampled_sentences": len(sentence_samples),
            "sampled_sentence_indices": [s["index"] for s in sentence_samples],
        },
        "sentence_evals": sentence_evals,
        "counts": counts,
        "total_comments": total_comments,
    }


def build_comparison(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    deltas = {}
    for level, cats in LEVEL_CATEGORIES.items():
        deltas[level] = {
            cat: int(b["counts"].get(level, {}).get(cat, 0)) - int(a["counts"].get(level, {}).get(cat, 0))
            for cat in cats
        }
    level_totals = {
        level: (
            sum(int(b["counts"].get(level, {}).get(cat, 0)) for cat in cats)
            - sum(int(a["counts"].get(level, {}).get(cat, 0)) for cat in cats)
        )
        for level, cats in LEVEL_CATEGORIES.items()
    }
    return {
        "baseline": a["file"],
        "candidate": b["file"],
        "baseline_counts": a["counts"],
        "candidate_counts": b["counts"],
        "delta_candidate_minus_baseline": deltas,
        "delta_candidate_minus_baseline_level_totals": level_totals,
        "baseline_total_comments": a["total_comments"],
        "candidate_total_comments": b["total_comments"],
    }


def parse_args() -> argparse.Namespace:
    default_provider = _normalize_provider_name(os.getenv("EVAL_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "openai")))
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate story quality comments from one or two txt files. "
            "The story is extracted from the last STORY START/END block."
        )
    )
    parser.add_argument("one", help="Path to first txt file")
    parser.add_argument("two", nargs="?", help="Optional second txt file to compare")
    parser.add_argument(
        "--provider",
        default=default_provider,
        choices=["openai", "gemini", "anthropic"],
        help="LLM provider (default: EVAL_LLM_PROVIDER/LLM_PROVIDER/openai)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("EVAL_LLM_MODEL"),
        help=(
            "Model name (default: EVAL_LLM_MODEL or the selected provider-specific default: "
            "OPENAI_MODEL/gpt-5.4-mini, GEMINI_MODEL/gemini-2.5-flash, "
            "ANTHROPIC_MODEL/claude-haiku-4-5)"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("EVAL_LLM_BASE_URL"),
        help=(
            "Optional API base URL override. Defaults to EVAL_LLM_BASE_URL or the provider-specific "
            "base URL from OPENAI_BASE_URL, GEMINI_BASE_URL, or ANTHROPIC_BASE_URL."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(os.getenv("EVAL_LLM_TIMEOUT_SECONDS", "600")),
        help="HTTP timeout for each LLM request in seconds (default: EVAL_LLM_TIMEOUT_SECONDS or 600)",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("EVAL_OUTPUT_PATH", "eval_output.txt"),
        help="Path to write evaluation JSON output (default: EVAL_OUTPUT_PATH or eval_output.txt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("EVAL_SEED", "0")),
        help="Random seed for chapter resampling when fewer than 5 chapters are available (default: EVAL_SEED or 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm = build_eval_llm(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
    )
    rng = random.Random(args.seed)

    first = evaluate_story_file(Path(args.one), llm, rng=rng)
    output: dict[str, Any] = {"one": first}

    if args.two:
        second = evaluate_story_file(Path(args.two), llm, rng=rng)
        output["two"] = second
        output["comparison"] = build_comparison(first, second)

    rendered = json.dumps(output, indent=2, ensure_ascii=False)
    print(rendered)
    Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
