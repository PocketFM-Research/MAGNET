from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


BLOCK_PATTERN = re.compile(
    r"=== USER PROMPT ===\s*(.*?)\s*=== LLM OUTPUT ===\s*(.*?)\s*=== END ===",
    flags=re.DOTALL,
)
STORY_BLOCK_PATTERN = re.compile(r"=== STORY START ===\s*(.*?)\s*=== STORY END ===", flags=re.DOTALL)
SCENE_HEADER_PATTERN = re.compile(r"===\s*SCENE\s+\d+\s*===")
FALLBACK_SCENE_WORD_TARGET = 2000


@dataclass
class ParagraphItem:
    text: str
    is_transition: bool


def extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text[start:], start=start):
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
                return text[start : i + 1]
    return None


def parse_log_paragraphs(raw_text: str) -> tuple[list[ParagraphItem], int]:
    items: list[ParagraphItem] = []
    skipped = 0

    for match in BLOCK_PATTERN.finditer(raw_text):
        user_prompt = match.group(1)
        llm_output = match.group(2).strip()

        if "TASK=narrate_step" in user_prompt:
            key = "paragraph"
            is_transition = False
        elif "TASK=new_goal" in user_prompt:
            key = "transition_paragraph"
            is_transition = True
        else:
            continue

        obj_text = extract_first_json_object(llm_output)
        if not obj_text:
            skipped += 1
            continue

        try:
            payload = json.loads(obj_text)
        except json.JSONDecodeError:
            skipped += 1
            continue

        paragraph = payload.get(key)
        if isinstance(paragraph, str) and paragraph.strip():
            items.append(ParagraphItem(text=paragraph.strip(), is_transition=is_transition))
        else:
            skipped += 1

    return items, skipped


def extract_story_text(raw_text: str) -> str | None:
    story_matches = list(STORY_BLOCK_PATTERN.finditer(raw_text))
    if story_matches:
        block = story_matches[-1].group(1).strip()
        obj_text = extract_first_json_object(block)
        if obj_text:
            try:
                payload = json.loads(obj_text)
                story_value = payload.get("story")
                if isinstance(story_value, str) and story_value.strip():
                    return story_value.strip()
            except json.JSONDecodeError:
                pass
        if block:
            return block

    if SCENE_HEADER_PATTERN.search(raw_text):
        return raw_text.strip()

    return None


def build_scenes(items: list[ParagraphItem]) -> list[str]:
    scenes: list[str] = []
    current_scene: list[str] = []

    for item in items:
        if item.is_transition:
            if current_scene:
                scenes.append("\n\n".join(current_scene))
            current_scene = [item.text]
        else:
            current_scene.append(item.text)

    if current_scene:
        scenes.append("\n\n".join(current_scene))

    return scenes


def split_story_text_into_scenes(story_text: str) -> list[str]:
    if SCENE_HEADER_PATTERN.search(story_text):
        chunks = re.split(r"(?=^===\s*SCENE\s+\d+\s*===\s*$)", story_text, flags=re.MULTILINE)
        scenes: list[str] = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            chunk = re.sub(r"^===\s*SCENE\s+\d+\s*===\s*\n?", "", chunk, count=1, flags=re.MULTILINE).strip()
            if chunk:
                scenes.append(chunk)
        return scenes

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", story_text) if p.strip()]
    if not paragraphs:
        return []

    normalized_paragraphs: list[str] = []
    for paragraph in paragraphs:
        paragraph_words = paragraph.split()
        if len(paragraph_words) <= FALLBACK_SCENE_WORD_TARGET:
            normalized_paragraphs.append(paragraph)
            continue

        sentence_parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
        if len(sentence_parts) <= 1:
            for i in range(0, len(paragraph_words), FALLBACK_SCENE_WORD_TARGET):
                normalized_paragraphs.append(" ".join(paragraph_words[i : i + FALLBACK_SCENE_WORD_TARGET]).strip())
            continue

        current_sentences: list[str] = []
        current_words = 0
        for sentence in sentence_parts:
            sentence_words = len(sentence.split())
            if current_sentences and current_words + sentence_words > FALLBACK_SCENE_WORD_TARGET:
                normalized_paragraphs.append(" ".join(current_sentences).strip())
                current_sentences = [sentence]
                current_words = sentence_words
            else:
                current_sentences.append(sentence)
                current_words += sentence_words
        if current_sentences:
            normalized_paragraphs.append(" ".join(current_sentences).strip())

    scenes: list[str] = []
    current_scene_paragraphs: list[str] = []
    current_scene_words = 0

    for paragraph in normalized_paragraphs:
        paragraph_words = len(paragraph.split())
        if (
            current_scene_paragraphs
            and current_scene_words + paragraph_words > FALLBACK_SCENE_WORD_TARGET
        ):
            scenes.append("\n\n".join(current_scene_paragraphs).strip())
            current_scene_paragraphs = [paragraph]
            current_scene_words = paragraph_words
        else:
            current_scene_paragraphs.append(paragraph)
            current_scene_words += paragraph_words

    if current_scene_paragraphs:
        scenes.append("\n\n".join(current_scene_paragraphs).strip())

    return [scene for scene in scenes if scene]


def format_scenes(scenes: list[str]) -> str:
    chunks: list[str] = []
    for i, scene in enumerate(scenes, start=1):
        chunks.append(f"=== SCENE {i} ===\n{scene}")
    return "\n\n".join(chunks).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split generated story logs into scenes based on transition paragraphs "
            "from TASK=new_goal blocks."
        )
    )
    parser.add_argument("input", help="Path to raw model output txt file")
    parser.add_argument(
        "--mode",
        choices=["auto", "log", "story"],
        default="auto",
        help=(
            "Parsing mode: auto (default) tries log transitions first, then story/scenes; "
            "log uses only TASK-based transitions; story uses STORY block or scene/fallback splitting."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path (default: <input>_scenes.txt)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    raw_text = in_path.read_text(encoding="utf-8")

    mode = ""
    skipped = 0
    items: list[ParagraphItem] = []
    scenes: list[str] = []
    if args.mode in {"auto", "log"}:
        items, skipped = parse_log_paragraphs(raw_text)
        scenes = build_scenes(items)
        mode = "log_transitions"
        if args.mode == "log":
            scenes = scenes
    if (args.mode == "story") or (args.mode == "auto" and not scenes):
        story_text = extract_story_text(raw_text)
        if story_text:
            scenes = split_story_text_into_scenes(story_text)
            mode = "story_block_or_scene_input"

    if not scenes:
        raise ValueError("No scenes could be extracted with the selected mode")

    out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_scenes.txt")
    out_path.write_text(format_scenes(scenes), encoding="utf-8")

    print(
        json.dumps(
            {
                "input_path": str(in_path),
                "output_path": str(out_path),
                "paragraphs_found": len(items),
                "scenes_written": len(scenes),
                "skipped_records": skipped,
                "mode": mode,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
