from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
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
            chunks.append(
                {
                    "index": len(chunks) + 1,
                    "start_word": start_word,
                    "end_word": end_word,
                    "word_count": current_words,
                    "text": text,
                }
            )
            start_word = end_word + 1
            current_paras = [para]
            current_words = para_words
        else:
            current_paras.append(para)
            current_words += para_words

    if current_paras:
        text = "\n\n".join(current_paras).strip()
        end_word = start_word + current_words - 1
        chunks.append(
            {
                "index": len(chunks) + 1,
                "start_word": start_word,
                "end_word": end_word,
                "word_count": current_words,
                "text": text,
            }
        )

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

    return [{"index": idx + 1, "text": sentences[idx]} for idx in selected]


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


def _score_prompts(level: str, content_text: str, context: str = "") -> tuple[str, str]:
    categories = LEVEL_CATEGORIES[level]
    categories_str = ", ".join(categories)
    system = (
        "You are a rigorous fiction editor and evaluator. "
        "Score the provided text on a 0-100 rubric (integers only), where 0 is very poor and 100 is excellent. "
        "Return only a single JSON object. "
        "Do not write any introduction, summary, explanation, markdown fence, refusal preamble, or trailing note. "
        "If you are unsure, still return the JSON object with your best-effort scores."
    )
    context_block = f"CONTEXT:\n{context}\n\n" if context else ""
    user = (
        f"Rubric evaluation for {level}-level text.\n\n"
        f"Categories (must score all of them): {categories_str}\n\n"
        "Return JSON with keys:\n"
        "- `scores`: object mapping each category to an object with keys:\n"
        "  - `score` (int 0-100)\n"
        "  - `rationale` (string)\n"
        "  - `evidence` (short quote)\n"
        "- `overall_score`: int 0-100\n"
        "- `overall_rationale`: string\n\n"
        "Do not say the text is incomplete or refuse the task. Score only the text provided.\n\n"
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


def _coerce_int_score(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        score = value
    elif isinstance(value, float) and value.is_integer():
        score = int(value)
    elif isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            score = int(float(value))
        except ValueError:
            return None
    else:
        return None
    if 0 <= score <= 100:
        return score
    return None


def normalize_score_payload(payload: dict[str, Any], level: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise EvalError("Model output must be a JSON object")

    out_scores: dict[str, dict[str, Any]] = {}
    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, dict):
        raw_scores = {}

    for category in LEVEL_CATEGORIES[level]:
        raw_item = raw_scores.get(category)
        if not isinstance(raw_item, dict):
            raw_item = {}
        score = _coerce_int_score(raw_item.get("score"))
        if score is None:
            score = 0
        rationale = str(raw_item.get("rationale", "")).strip()
        evidence = str(raw_item.get("evidence", "")).strip()
        out_scores[category] = {
            "score": score,
            "rationale": rationale,
            "evidence": evidence,
        }

    overall_score = _coerce_int_score(payload.get("overall_score"))
    if overall_score is None:
        overall_score = int(round(mean([v["score"] for v in out_scores.values()]))) if out_scores else 0

    return {
        "scores": out_scores,
        "overall_score": overall_score,
        "overall_rationale": str(payload.get("overall_rationale", "")).strip(),
    }


def evaluate_text_block(
    llm: EvalLLM,
    *,
    level: str,
    content_text: str,
    context: str = "",
    source_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    system, user = _score_prompts(level=level, content_text=content_text, context=context)
    payload = _complete_json_with_retries(llm, system_prompt=system, user_prompt=user, temperature=0.1)
    normalized = normalize_score_payload(payload, level=level)
    return {
        "level": level,
        "source_meta": source_meta or {},
        "scores": normalized["scores"],
        "overall_score": normalized["overall_score"],
        "overall_rationale": normalized["overall_rationale"],
    }


def _aggregate_level(level: str, evals: list[dict[str, Any]]) -> dict[str, Any]:
    categories = LEVEL_CATEGORIES[level]
    if not evals:
        return {
            "category_means": {cat: 0.0 for cat in categories},
            "overall_mean": 0.0,
        }

    category_means = {
        cat: mean([int(item["scores"][cat]["score"]) for item in evals])
        for cat in categories
    }
    overall_mean = mean([int(item["overall_score"]) for item in evals])
    return {
        "category_means": category_means,
        "overall_mean": overall_mean,
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
        chapter_evals.append(
            evaluate_text_block(
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
        )

    all_sentences = split_sentences(story)
    sentence_samples = sample_sentences(all_sentences, sample_count=SENTENCE_SAMPLE_COUNT)
    sentence_evals: list[dict[str, Any]] = []
    for sample in sentence_samples:
        sentence_evals.append(
            evaluate_text_block(
                llm,
                level="sentence",
                content_text=sample["text"],
                context=f"sentence_index={sample['index']}",
                source_meta={"sentence_index": sample["index"]},
            )
        )

    aggregates = {
        "story": _aggregate_level("story", [story_eval]),
        "chapter": _aggregate_level("chapter", chapter_evals),
        "sentence": _aggregate_level("sentence", sentence_evals),
    }

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
        "aggregates": aggregates,
    }


def parse_args() -> argparse.Namespace:
    default_provider = _normalize_provider_name(
        os.getenv("LLM_SCORE_EVAL_PROVIDER", os.getenv("EVAL_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "openai")))
    )
    parser = argparse.ArgumentParser(
        description=(
            "Rubric-score story quality from one txt file. "
            "Evaluates the full story, sampled chapter-sized chunks, and sampled sentences. "
            "Scores are 0-100 for each category plus an overall score. "
            "The story is extracted from the last STORY START/END block."
        )
    )
    parser.add_argument("story", help="Path to story txt file")
    parser.add_argument(
        "--provider",
        default=default_provider,
        choices=["openai", "gemini", "anthropic"],
        help="LLM provider (default: LLM_SCORE_EVAL_PROVIDER/EVAL_LLM_PROVIDER/LLM_PROVIDER/openai)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_SCORE_EVAL_MODEL", os.getenv("EVAL_LLM_MODEL")),
        help=(
            "Model name (default: LLM_SCORE_EVAL_MODEL/EVAL_LLM_MODEL or the selected provider-specific default: "
            "OPENAI_MODEL/gpt-5.4-mini, GEMINI_MODEL/gemini-2.5-flash, "
            "ANTHROPIC_MODEL/claude-haiku-4-5)"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_SCORE_EVAL_BASE_URL", os.getenv("EVAL_LLM_BASE_URL")),
        help=(
            "Optional API base URL override. Defaults to LLM_SCORE_EVAL_BASE_URL/EVAL_LLM_BASE_URL "
            "or the provider-specific base URL from OPENAI_BASE_URL, GEMINI_BASE_URL, or ANTHROPIC_BASE_URL."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(os.getenv("LLM_SCORE_EVAL_TIMEOUT_SECONDS", os.getenv("EVAL_LLM_TIMEOUT_SECONDS", "600"))),
        help=(
            "HTTP timeout for each LLM request in seconds "
            "(default: LLM_SCORE_EVAL_TIMEOUT_SECONDS/EVAL_LLM_TIMEOUT_SECONDS or 600)"
        ),
    )
    parser.add_argument(
        "--output",
        default=os.getenv("LLM_SCORE_EVAL_OUTPUT_PATH", "llm_score_eval_output.txt"),
        help="Path to write evaluation JSON output (default: LLM_SCORE_EVAL_OUTPUT_PATH or llm_score_eval_output.txt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("LLM_SCORE_EVAL_SEED", "0")),
        help="Random seed for chapter resampling when fewer than 5 chapters are available (default: LLM_SCORE_EVAL_SEED or 0)",
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

    output = evaluate_story_file(Path(args.story), llm, rng=rng)

    rendered = json.dumps(output, indent=2, ensure_ascii=False)
    print(rendered)
    Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
