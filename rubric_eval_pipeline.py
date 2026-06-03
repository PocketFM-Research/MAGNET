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
        out.append(
            {
                "index": idx + 1,
                "text": sentences[idx],
            }
        )
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


def _rubric_prompts(level: str, presented_texts: dict[str, str], context: str = "") -> tuple[str, str]:
    categories = LEVEL_CATEGORIES[level]
    categories_str = ", ".join(categories)
    labels = list(presented_texts.keys())
    if labels != ["A", "B"] and labels != ["A", "B", "C"]:
        raise EvalError("Presented labels must be ['A', 'B'] or ['A', 'B', 'C']")

    versions_word = "TWO" if len(labels) == 2 else "THREE"
    labels_str = " and ".join(labels) if len(labels) == 2 else ", ".join(labels[:-1]) + f", and {labels[-1]}"
    winner_options = ", ".join([f"\"{label}\"" for label in labels] + ["\"tie\""])
    side_keys = "\n".join([f"- `{label}`: object with keys `scores` and `overall_score`" for label in labels])

    system = (
        "You are a rigorous fiction editor and evaluator. "
        f"You will compare {versions_word} versions of text labeled {labels_str}. "
        "Score each category on a 0-100 rubric (integers only), where 0 is very poor and 100 is excellent. "
        f"Return only JSON. Refer only to the labels {labels_str}."
    )
    context_block = f"CONTEXT:\n{context}\n\n" if context else ""
    text_blocks = "\n\n".join([f"{label}_TEXT:\n{presented_texts[label]}" for label in labels])
    user = (
        f"Rubric evaluation for {level}-level text.\n\n"
        f"Categories (must score all of them): {categories_str}\n\n"
        "Return JSON with keys:\n"
        f"{side_keys}\n"
        f"- `winner`: one of {winner_options}\n"
        "- `winner_rationale`: string explaining why the winner is better\n\n"
        "Where each side's `scores` is an object mapping each category to an object with keys:\n"
        "- `score` (int 0-100)\n"
        "- `rationale` (string)\n"
        "- `evidence` (short quote)\n\n"
        f"{context_block}"
        f"{text_blocks}"
    )
    return system, user


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


def _normalize_side(side_payload: Any, level: str) -> dict[str, Any]:
    if not isinstance(side_payload, dict):
        side_payload = {}

    out_scores: dict[str, dict[str, Any]] = {}
    raw_scores = side_payload.get("scores")
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

    overall_score = _coerce_int_score(side_payload.get("overall_score"))
    if overall_score is None:
        overall_score = int(round(mean([v["score"] for v in out_scores.values()]))) if out_scores else 0

    return {"scores": out_scores, "overall_score": overall_score}


def normalize_rubric(payload: dict[str, Any], level: str, labels: list[str]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise EvalError("Model output must be a JSON object")

    normalized_sides = {label: _normalize_side(payload.get(label), level=level) for label in labels}

    winner = str(payload.get("winner", "")).strip().lower()
    valid_winners = {label.lower() for label in labels} | {"tie"}
    if winner not in valid_winners:
        winner = "tie"
    winner_rationale = str(payload.get("winner_rationale", "")).strip()

    return {**normalized_sides, "winner": winner, "winner_rationale": winner_rationale}


def evaluate_text_group(
    llm: OpenAILLM,
    *,
    level: str,
    texts_by_story: dict[str, str],
    context: str = "",
    source_meta: dict[str, Any] | None = None,
    rng: random.Random,
) -> dict[str, Any]:
    story_keys = list(texts_by_story.keys())
    display_labels = ["A", "B", "C"][: len(story_keys)]
    shuffled_story_keys = list(story_keys)
    rng.shuffle(shuffled_story_keys)
    presentation_order = dict(zip(display_labels, shuffled_story_keys))
    presented_texts = {label: texts_by_story[story_key] for label, story_key in presentation_order.items()}

    system, user = _rubric_prompts(level=level, presented_texts=presented_texts, context=context)
    payload = llm.complete_json(system_prompt=system, user_prompt=user, temperature=0.1)
    normalized = normalize_rubric(payload, level=level, labels=display_labels)

    results_by_story = {
        story_key: normalized[label]
        for label, story_key in presentation_order.items()
    }

    winner_label = normalized["winner"]
    winner_story = presentation_order[winner_label.upper()] if winner_label != "tie" else "tie"

    return {
        "level": level,
        "source_meta": source_meta or {},
        "presentation_order": presentation_order,
        "results": results_by_story,
        "winner": winner_story,
        "winner_rationale": normalized.get("winner_rationale", ""),
    }


def _aggregate_level(level: str, evals: list[dict[str, Any]], story_key: str) -> dict[str, Any]:
    categories = LEVEL_CATEGORIES[level]
    if not evals:
        return {
            "category_means": {cat: 0.0 for cat in categories},
            "overall_mean": 0.0,
        }

    category_means: dict[str, float] = {}
    for cat in categories:
        category_means[cat] = mean([int(item["results"][story_key]["scores"][cat]["score"]) for item in evals])
    overall_mean = mean([int(item["results"][story_key]["overall_score"]) for item in evals])
    return {"category_means": category_means, "overall_mean": overall_mean}


def _read_story(path: Path) -> str:
    return extract_story_block(path.read_text(encoding="utf-8"))


def evaluate_story_set(
    story_paths: list[Path],
    llm: OpenAILLM,
    *,
    seed: int | None,
) -> dict[str, Any]:
    if len(story_paths) not in {2, 3}:
        raise EvalError("Expected 2 or 3 story paths")

    rng = random.Random(seed)
    story_keys = [f"story_{idx + 1}" for idx in range(len(story_paths))]
    stories = {story_key: _read_story(path) for story_key, path in zip(story_keys, story_paths)}

    story_eval = evaluate_text_group(
        llm,
        level="story",
        texts_by_story=stories,
        source_meta={"unit": "full_story"},
        rng=rng,
    )

    chapters_by_story = {
        story_key: chunk_story_by_words(story_text, target_words=CHAPTER_WORD_TARGET)
        for story_key, story_text in stories.items()
    }
    sampled_chapters_by_story = {
        story_key: sample_chapters(chapters, sample_count=CHAPTER_SAMPLE_COUNT, rng=rng)
        for story_key, chapters in chapters_by_story.items()
    }

    num_chapter_groups = min(len(sampled) for sampled in sampled_chapters_by_story.values())
    chapter_evals: list[dict[str, Any]] = []
    for i in range(num_chapter_groups):
        chapter_group = {
            story_key: sampled_chapters_by_story[story_key][i]
            for story_key in story_keys
        }
        group_eval = evaluate_text_group(
            llm,
            level="chapter",
            texts_by_story={story_key: chapter["text"] for story_key, chapter in chapter_group.items()},
            context="; ".join(
                [
                    (
                        f"{story_key}_chapter_index={chapter_group[story_key]['index']}, "
                        f"{story_key}_word_span={chapter_group[story_key]['start_word']}-{chapter_group[story_key]['end_word']}"
                    )
                    for story_key in story_keys
                ]
            ),
            source_meta={
                "group_index": i + 1,
                "stories": {
                    story_key: {
                        "chapter_index": chapter_group[story_key]["index"],
                        "start_word": chapter_group[story_key]["start_word"],
                        "end_word": chapter_group[story_key]["end_word"],
                        "word_count": chapter_group[story_key]["word_count"],
                    }
                    for story_key in story_keys
                },
            },
            rng=rng,
        )
        chapter_evals.append(group_eval)

    sentences_by_story = {
        story_key: split_sentences(story_text)
        for story_key, story_text in stories.items()
    }
    sampled_sentences_by_story = {
        story_key: sample_sentences(sentences, sample_count=SENTENCE_SAMPLE_COUNT)
        for story_key, sentences in sentences_by_story.items()
    }

    num_sentence_groups = min(len(sampled) for sampled in sampled_sentences_by_story.values())
    sentence_evals: list[dict[str, Any]] = []
    for i in range(num_sentence_groups):
        sentence_group = {
            story_key: sampled_sentences_by_story[story_key][i]
            for story_key in story_keys
        }
        group_eval = evaluate_text_group(
            llm,
            level="sentence",
            texts_by_story={story_key: sentence["text"] for story_key, sentence in sentence_group.items()},
            context="; ".join(
                [f"{story_key}_sentence_index={sentence_group[story_key]['index']}" for story_key in story_keys]
            ),
            source_meta={
                "group_index": i + 1,
                "stories": {
                    story_key: {"sentence_index": sentence_group[story_key]["index"]}
                    for story_key in story_keys
                },
            },
            rng=rng,
        )
        sentence_evals.append(group_eval)

    aggregates = {
        story_key: {
            "story": _aggregate_level("story", [story_eval], story_key=story_key),
            "chapter": _aggregate_level("chapter", chapter_evals, story_key=story_key),
            "sentence": _aggregate_level("sentence", sentence_evals, story_key=story_key),
        }
        for story_key in story_keys
    }

    return {
        "story_files": {
            story_key: str(path)
            for story_key, path in zip(story_keys, story_paths)
        },
        "seed": seed,
        "level_categories": LEVEL_CATEGORIES,
        "plans": {
            "chapter": {
                "target_words": CHAPTER_WORD_TARGET,
                "sample_count_target": CHAPTER_SAMPLE_COUNT,
                "num_groups": len(chapter_evals),
                "stories": {
                    story_key: {
                        "num_chapters_total": len(chapters_by_story[story_key]),
                        "sampled_chapter_indices": [c["index"] for c in sampled_chapters_by_story[story_key]],
                    }
                    for story_key in story_keys
                },
            },
            "sentence": {
                "sample_count_target": SENTENCE_SAMPLE_COUNT,
                "num_groups": len(sentence_evals),
                "stories": {
                    story_key: {
                        "num_sentences_total": len(sentences_by_story[story_key]),
                        "sampled_sentence_indices": [s["index"] for s in sampled_sentences_by_story[story_key]],
                    }
                    for story_key in story_keys
                },
            },
        },
        "group_evals": {
            "story": story_eval,
            "chapters": chapter_evals,
            "sentences": sentence_evals,
        },
        "aggregates": aggregates,
    }


def build_comparison(summary: dict[str, Any]) -> dict[str, Any]:
    story_files = summary.get("story_files", {})
    aggregates = summary.get("aggregates", {})

    def get_means(story_key: str, level: str) -> dict[str, float]:
        agg = aggregates.get(story_key, {}).get(level, {})
        means = agg.get("category_means", {})
        if not isinstance(means, dict):
            means = {}
        return {cat: float(means.get(cat, 0.0)) for cat in LEVEL_CATEGORIES[level]}

    story_keys = list(story_files.keys())
    rankings: dict[str, Any] = {}
    for level in CATEGORY_ORDER:
        rankings[level] = sorted(
            [
                {
                    "story_key": story_key,
                    "file": story_files.get(story_key),
                    "overall_mean": float(aggregates.get(story_key, {}).get(level, {}).get("overall_mean", 0.0)),
                    "category_means": get_means(story_key, level),
                }
                for story_key in story_keys
            ],
            key=lambda item: item["overall_mean"],
            reverse=True,
        )

    return {
        "story_files": story_files,
        "rankings": rankings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rubric-evaluate story quality from two or three txt files. "
            "Each evaluation call compares all provided stories together in a single pass. "
            "Randomizes which version is shown first (A/B or A/B/C) per evaluation. "
            "Scores are 0-100 for each category (per version), plus an overall score. "
            "The story is extracted from the last STORY START/END block."
        )
    )
    parser.add_argument(
        "stories",
        nargs="+",
        help="Paths to 2 or 3 story txt files",
    )
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
        default=os.getenv("RUBRIC_EVAL_OUTPUT_PATH", "rubric_eval_output.txt"),
        help="Path to write evaluation JSON output (default: RUBRIC_EVAL_OUTPUT_PATH or rubric_eval_output.txt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("RUBRIC_EVAL_SEED", "0")),
        help="Random seed for A/B presentation order (default: RUBRIC_EVAL_SEED or 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EvalError("OPENAI_API_KEY is required")
    if len(args.stories) not in {2, 3}:
        raise EvalError("Provide exactly 2 or 3 story files")

    llm = OpenAILLM(api_key=api_key, model=args.model, base_url=args.base_url)

    output = evaluate_story_set([Path(story) for story in args.stories], llm, seed=args.seed)

    rendered = json.dumps(output, indent=2, ensure_ascii=False)
    print(rendered)
    Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
