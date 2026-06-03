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
        "Return only JSON. "
        "Each comment must be a specific, actionable critique tied to the provided text. "
        f"Allowed categories: {categories}."
    )
    context_block = f"CONTEXT:\n{context}\n\n" if context else ""
    user = (
        f"Read the {level}-level text and annotate editor comments. "
        "Include up to 100 comments. "
        "Return JSON with key `comments`, where `comments` is an array of objects with keys: "
        "`category` (one allowed category), `comment` (string), `evidence` (short quote or reference). "
        f"{context_block}"
        f"TEXT:\n{content_text}"
    )
    return system, user


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
    llm: OpenAILLM,
    *,
    level: str,
    content_text: str,
    context: str = "",
    source_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    system, user = _editor_prompts(level=level, content_text=content_text, context=context)
    payload = llm.complete_json(system_prompt=system, user_prompt=user, temperature=0.1)
    comments = normalize_comments(payload, level=level)
    counts = count_by_category(comments, LEVEL_CATEGORIES[level])
    return {
        "level": level,
        "source_meta": source_meta or {},
        "comments": comments,
        "counts": counts,
        "total_comments": len(comments),
    }


def evaluate_story_file(path: Path, llm: OpenAILLM, *, rng: random.Random) -> dict[str, Any]:
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
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("EVAL_SEED", "0")),
        help="Random seed for chapter resampling when fewer than 5 chapters are available (default: EVAL_SEED or 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EvalError("OPENAI_API_KEY is required")

    llm = OpenAILLM(api_key=api_key, model=args.model, base_url=args.base_url)
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
