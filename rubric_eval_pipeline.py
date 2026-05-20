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
    paras = [p.strip() for p in re.split(r"\n\s*\n", story) if p.strip()]
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

    out: list[dict[str, Any]] = []
    for idx in selected:
        out.append(
            {
                "index": idx + 1,
                "text": sentences[idx],
            }
        )
    return out


def sample_chapters(chapters: list[dict[str, Any]], sample_count: int = CHAPTER_SAMPLE_COUNT) -> list[dict[str, Any]]:
    if not chapters:
        return []
    if len(chapters) <= sample_count:
        selected = list(range(len(chapters)))
    else:
        selected = sorted({round(i * (len(chapters) - 1) / (sample_count - 1)) for i in range(sample_count)})
    return [chapters[idx] for idx in selected]


def _pairwise_rubric_prompts(level: str, a_text: str, b_text: str, context: str = "") -> tuple[str, str]:
    categories = LEVEL_CATEGORIES[level]
    categories_str = ", ".join(categories)
    system = (
        "You are a rigorous fiction editor and evaluator. "
        "You will compare TWO versions of text labeled A and B. "
        "Score each category on a 0-100 rubric (integers only), where 0 is very poor and 100 is excellent. "
        "Return only JSON. Do not mention which is 'baseline' or 'candidate'—only refer to A and B."
    )
    context_block = f"CONTEXT:\n{context}\n\n" if context else ""
    user = (
        f"Pairwise rubric evaluation for {level}-level text.\n\n"
        f"Categories (must score all of them): {categories_str}\n\n"
        "Return JSON with keys:\n"
        "- `A`: object with keys `scores` and `overall_score`\n"
        "- `B`: object with keys `scores` and `overall_score`\n"
        "- `winner`: one of \"A\", \"B\", or \"tie\"\n"
        "- `winner_rationale`: string explaining why the winner is better\n\n"
        "Where each side's `scores` is an object mapping each category to an object with keys:\n"
        "- `score` (int 0-100)\n"
        "- `rationale` (string)\n"
        "- `evidence` (short quote)\n\n"
        f"{context_block}"
        f"A_TEXT:\n{a_text}\n\n"
        f"B_TEXT:\n{b_text}"
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


def normalize_pairwise_rubric(payload: dict[str, Any], level: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise EvalError("Model output must be a JSON object")

    a_norm = _normalize_side(payload.get("A"), level=level)
    b_norm = _normalize_side(payload.get("B"), level=level)

    winner = str(payload.get("winner", "")).strip().lower()
    if winner not in {"a", "b", "tie"}:
        winner = "tie"
    winner_rationale = str(payload.get("winner_rationale", "")).strip()

    return {
        "A": a_norm,
        "B": b_norm,
        "winner": winner,
        "winner_rationale": winner_rationale,
    }


def evaluate_pair_text_block(
    llm: OpenAILLM,
    *,
    level: str,
    baseline_text: str,
    candidate_text: str,
    context: str = "",
    source_meta: dict[str, Any] | None = None,
    rng: random.Random,
) -> dict[str, Any]:
    if rng.random() < 0.5:
        a_label = "baseline"
        b_label = "candidate"
        a_text = baseline_text
        b_text = candidate_text
    else:
        a_label = "candidate"
        b_label = "baseline"
        a_text = candidate_text
        b_text = baseline_text

    system, user = _pairwise_rubric_prompts(level=level, a_text=a_text, b_text=b_text, context=context)
    payload = llm.complete_json(system_prompt=system, user_prompt=user, temperature=0.1)
    normalized = normalize_pairwise_rubric(payload, level=level)

    a_side = normalized["A"]
    b_side = normalized["B"]
    baseline_out = a_side if a_label == "baseline" else b_side
    candidate_out = b_side if b_label == "candidate" else a_side

    winner = normalized["winner"]
    if winner == "a":
        winner_abs = a_label
    elif winner == "b":
        winner_abs = b_label
    else:
        winner_abs = "tie"

    return {
        "level": level,
        "source_meta": source_meta or {},
        "presentation_order": {"A": a_label, "B": b_label},
        "baseline": baseline_out,
        "candidate": candidate_out,
        "winner": winner_abs,
        "winner_rationale": normalized.get("winner_rationale", ""),
    }


def _aggregate_level(level: str, evals: list[dict[str, Any]], side: str) -> dict[str, Any]:
    categories = LEVEL_CATEGORIES[level]
    if not evals:
        return {
            "category_means": {cat: 0.0 for cat in categories},
            "overall_mean": 0.0,
        }

    category_means: dict[str, float] = {}
    for cat in categories:
        category_means[cat] = mean([int(item[side]["scores"][cat]["score"]) for item in evals])
    overall_mean = mean([int(item[side]["overall_score"]) for item in evals])
    return {"category_means": category_means, "overall_mean": overall_mean}


def _read_story(path: Path) -> str:
    return extract_story_block(path.read_text(encoding="utf-8"))


def evaluate_story_pair(
    baseline_path: Path,
    candidate_path: Path,
    llm: OpenAILLM,
    *,
    seed: int | None,
) -> dict[str, Any]:
    rng = random.Random(seed)

    baseline_story = _read_story(baseline_path)
    candidate_story = _read_story(candidate_path)

    story_eval = evaluate_pair_text_block(
        llm,
        level="story",
        baseline_text=baseline_story,
        candidate_text=candidate_story,
        source_meta={"unit": "full_story"},
        rng=rng,
    )

    baseline_chapters = chunk_story_by_words(baseline_story, target_words=CHAPTER_WORD_TARGET)
    candidate_chapters = chunk_story_by_words(candidate_story, target_words=CHAPTER_WORD_TARGET)
    baseline_sampled = sample_chapters(baseline_chapters, sample_count=CHAPTER_SAMPLE_COUNT)
    candidate_sampled = sample_chapters(candidate_chapters, sample_count=CHAPTER_SAMPLE_COUNT)

    num_chapter_pairs = min(len(baseline_sampled), len(candidate_sampled))
    chapter_pairs: list[dict[str, Any]] = []
    for i in range(num_chapter_pairs):
        b_ch = baseline_sampled[i]
        c_ch = candidate_sampled[i]
        pair_eval = evaluate_pair_text_block(
            llm,
            level="chapter",
            baseline_text=b_ch["text"],
            candidate_text=c_ch["text"],
            context=(
                f"baseline_chapter_index={b_ch['index']}, baseline_word_span={b_ch['start_word']}-{b_ch['end_word']}; "
                f"candidate_chapter_index={c_ch['index']}, candidate_word_span={c_ch['start_word']}-{c_ch['end_word']}"
            ),
            source_meta={
                "pair_index": i + 1,
                "baseline": {
                    "chapter_index": b_ch["index"],
                    "start_word": b_ch["start_word"],
                    "end_word": b_ch["end_word"],
                    "word_count": b_ch["word_count"],
                },
                "candidate": {
                    "chapter_index": c_ch["index"],
                    "start_word": c_ch["start_word"],
                    "end_word": c_ch["end_word"],
                    "word_count": c_ch["word_count"],
                },
            },
            rng=rng,
        )
        chapter_pairs.append(pair_eval)

    baseline_sentences = split_sentences(baseline_story)
    candidate_sentences = split_sentences(candidate_story)
    baseline_sentence_samples = sample_sentences(baseline_sentences, sample_count=SENTENCE_SAMPLE_COUNT)
    candidate_sentence_samples = sample_sentences(candidate_sentences, sample_count=SENTENCE_SAMPLE_COUNT)

    num_sentence_pairs = min(len(baseline_sentence_samples), len(candidate_sentence_samples))
    sentence_pairs: list[dict[str, Any]] = []
    for i in range(num_sentence_pairs):
        b_s = baseline_sentence_samples[i]
        c_s = candidate_sentence_samples[i]
        pair_eval = evaluate_pair_text_block(
            llm,
            level="sentence",
            baseline_text=b_s["text"],
            candidate_text=c_s["text"],
            context=f"baseline_sentence_index={b_s['index']}; candidate_sentence_index={c_s['index']}",
            source_meta={
                "pair_index": i + 1,
                "baseline": {"sentence_index": b_s["index"]},
                "candidate": {"sentence_index": c_s["index"]},
            },
            rng=rng,
        )
        sentence_pairs.append(pair_eval)

    aggregates = {
        "baseline": {
            "story": _aggregate_level("story", [story_eval], side="baseline"),
            "chapter": _aggregate_level("chapter", chapter_pairs, side="baseline"),
            "sentence": _aggregate_level("sentence", sentence_pairs, side="baseline"),
        },
        "candidate": {
            "story": _aggregate_level("story", [story_eval], side="candidate"),
            "chapter": _aggregate_level("chapter", chapter_pairs, side="candidate"),
            "sentence": _aggregate_level("sentence", sentence_pairs, side="candidate"),
        },
    }

    return {
        "baseline_file": str(baseline_path),
        "candidate_file": str(candidate_path),
        "seed": seed,
        "level_categories": LEVEL_CATEGORIES,
        "plans": {
            "chapter": {
                "target_words": CHAPTER_WORD_TARGET,
                "sample_count_target": CHAPTER_SAMPLE_COUNT,
                "baseline_num_chapters_total": len(baseline_chapters),
                "candidate_num_chapters_total": len(candidate_chapters),
                "num_pairs": len(chapter_pairs),
                "baseline_sampled_chapter_indices": [c["index"] for c in baseline_sampled],
                "candidate_sampled_chapter_indices": [c["index"] for c in candidate_sampled],
            },
            "sentence": {
                "sample_count_target": SENTENCE_SAMPLE_COUNT,
                "baseline_num_sentences_total": len(baseline_sentences),
                "candidate_num_sentences_total": len(candidate_sentences),
                "num_pairs": len(sentence_pairs),
                "baseline_sampled_sentence_indices": [s["index"] for s in baseline_sentence_samples],
                "candidate_sampled_sentence_indices": [s["index"] for s in candidate_sentence_samples],
            },
        },
        "pairwise_evals": {
            "story": story_eval,
            "chapters": chapter_pairs,
            "sentences": sentence_pairs,
        },
        "aggregates": aggregates,
    }


def build_comparison(pair: dict[str, Any]) -> dict[str, Any]:
    def get_means(side: str, level: str) -> dict[str, float]:
        agg = pair.get("aggregates", {}).get(side, {}).get(level, {})
        means = agg.get("category_means", {})
        if not isinstance(means, dict):
            means = {}
        return {cat: float(means.get(cat, 0.0)) for cat in LEVEL_CATEGORIES[level]}

    deltas: dict[str, Any] = {}
    for level in CATEGORY_ORDER:
        base_means = get_means("baseline", level)
        cand_means = get_means("candidate", level)
        deltas[level] = {cat: cand_means[cat] - base_means[cat] for cat in LEVEL_CATEGORIES[level]}

    return {
        "baseline": pair.get("baseline_file"),
        "candidate": pair.get("candidate_file"),
        "delta_candidate_minus_baseline_category_means": deltas,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pairwise rubric-evaluate story quality from two txt files. "
            "Randomizes which version is shown first (A vs B) per evaluation. "
            "Scores are 0-100 for each category (per version), plus an overall score. "
            "The story is extracted from the last STORY START/END block."
        )
    )
    parser.add_argument("baseline", help="Path to baseline txt file")
    parser.add_argument("candidate", help="Path to candidate/framework txt file")
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

    llm = OpenAILLM(api_key=api_key, model=args.model, base_url=args.base_url)

    pair = evaluate_story_pair(Path(args.baseline), Path(args.candidate), llm, seed=args.seed)
    output: dict[str, Any] = {"pairwise": pair, "comparison": build_comparison(pair)}

    rendered = json.dumps(output, indent=2, ensure_ascii=False)
    print(rendered)
    Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
