from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from environment import WorldProxyEnv
from fables import FableDefinition
from llm import LLMError, _parse_json_object_with_repair
from sim_types import CharacterDecision, RecoveredRunState


@dataclass
class LogBlock:
    timestamp: str
    model: str
    system_prompt: str
    user_prompt: str
    llm_output: str
    task_name: str


_BLOCK_PATTERN = re.compile(
    r"\[(?P<timestamp>[^\]]+)\]\s+model=(?P<model>[^\n]+)\n"
    r"=== SYSTEM PROMPT ===\n(?P<system>.*?)\n"
    r"=== USER PROMPT ===\n(?P<user>.*?)\n"
    r"=== LLM OUTPUT ===\n(?P<output>.*?)\n"
    r"=== END ===\n",
    flags=re.DOTALL,
)


def recover_state_from_llm_output(log_path: str | Path, fable: FableDefinition) -> RecoveredRunState:
    blocks = _parse_blocks(Path(log_path))
    if not blocks:
        raise ValueError(f"No complete LLM log blocks found in {log_path}.")

    sessions = _split_sessions(blocks)
    latest_session = sessions[-1] if sessions else blocks

    env = WorldProxyEnv(fable=fable)
    characters = [character.name for character in fable.characters]
    env.reset(characters)

    story: list[str] = []
    timeline: list[str] = [f"recovered_from_log path={Path(log_path)}"]
    total_reward = 0.0
    goal_assigned_step = 1
    recovered_turns = 0
    last_completed_step = 0

    for block in latest_session:
        if block.task_name == "narrate_step":
            recovered = _apply_narrate_step_block(
                env=env,
                block=block,
                story=story,
                timeline=timeline,
            )
            if recovered is None:
                continue
            total_reward += recovered["step_reward"]
            recovered_turns += 1
            last_completed_step = recovered["step_number"]
        elif block.task_name == "new_goal":
            applied = _apply_new_goal_block(
                env=env,
                block=block,
                story=story,
                timeline=timeline,
            )
            if applied and last_completed_step > 0:
                goal_assigned_step = last_completed_step + 1

    world_vars = env.get_world_vars()
    next_step = int(world_vars.get("turn", 0)) + 1
    timeline.append(
        f"recovery_complete recovered_turns={recovered_turns} next_step={next_step} current_goal={env.get_current_goal()}"
    )
    return RecoveredRunState(
        world_vars=world_vars,
        story=story,
        timeline=timeline,
        total_reward=total_reward,
        next_step=next_step,
        goal_assigned_step=goal_assigned_step,
        source_path=str(Path(log_path)),
        recovered_turns=recovered_turns,
    )


def _parse_blocks(path: Path) -> list[LogBlock]:
    raw = path.read_text(encoding="utf-8")
    blocks: list[LogBlock] = []
    for match in _BLOCK_PATTERN.finditer(raw):
        user_prompt = match.group("user")
        task_match = re.search(r"^\s*TASK=([A-Za-z0-9_:-]+)", user_prompt, flags=re.MULTILINE)
        blocks.append(
            LogBlock(
                timestamp=match.group("timestamp"),
                model=match.group("model").strip(),
                system_prompt=match.group("system"),
                user_prompt=user_prompt,
                llm_output=match.group("output"),
                task_name=task_match.group(1).strip().lower() if task_match else "",
            )
        )
    return blocks


def _split_sessions(blocks: list[LogBlock]) -> list[list[LogBlock]]:
    sessions: list[list[LogBlock]] = []
    current: list[LogBlock] = []
    for block in blocks:
        if _starts_new_session(block) and current:
            sessions.append(current)
            current = []
        current.append(block)
    if current:
        sessions.append(current)
    return sessions


def _starts_new_session(block: LogBlock) -> bool:
    if block.task_name != "narrate_step":
        return False
    recent_story = _extract_json_field(block.user_prompt, "Recent story paragraphs")
    world_before = _extract_json_field(block.user_prompt, "World before")
    return recent_story == [] or (isinstance(world_before, dict) and int(world_before.get("turn", -1)) == 0)


def _apply_narrate_step_block(
    env: WorldProxyEnv,
    block: LogBlock,
    story: list[str],
    timeline: list[str],
) -> dict[str, Any] | None:
    world_before = _extract_json_field(block.user_prompt, "World before")
    proposals = _extract_json_field(block.user_prompt, "Proposed actions")
    if not isinstance(world_before, dict) or not isinstance(proposals, list):
        return None

    parsed_output = _safe_parse_output(block.llm_output)
    if not isinstance(parsed_output, dict):
        return None

    included_indices = parsed_output.get("included_indices", [])
    paragraph = str(parsed_output.get("paragraph", "")).strip()
    if not isinstance(included_indices, list) or not paragraph:
        return None

    character_names = world_before.get("characters")
    if not isinstance(character_names, list) or not all(isinstance(name, str) for name in character_names):
        return None

    env.restore_world_vars(character_names, deepcopy(world_before))
    selected_actions: list[CharacterDecision] = []
    for index in included_indices:
        if not isinstance(index, int) or not (0 <= index < len(proposals)):
            continue
        proposal = proposals[index]
        if not isinstance(proposal, dict):
            continue
        selected_actions.append(
            CharacterDecision(
                character=str(proposal.get("character", "")).strip(),
                action=str(proposal.get("action", "")).strip(),
                confidence=_coerce_float(proposal.get("confidence", 0.0)),
                revisions_used=0,
                rationale=str(proposal.get("rationale", "")).strip(),
                advances_goal=bool(proposal.get("advances_goal", False)),
                goal_reached=bool(proposal.get("goal_reached", False)),
                world_updates=proposal.get("world_updates", {}) if isinstance(proposal.get("world_updates", {}), dict) else {},
                progress_reason=str(proposal.get("progress_reason", "")).strip(),
            )
        )

    if not selected_actions:
        return None

    results = env.step_selected_actions(selected_actions)
    step_reward = sum(result.reward for result in results)
    step_number = int(env.get_world_vars().get("turn", 0))
    story[:] = _merged_story_prefix(story, block.user_prompt)
    story.append(paragraph)
    timeline.append(f"recovered_t={step_number} story={paragraph}")
    return {"step_reward": step_reward, "step_number": step_number}


def _apply_new_goal_block(
    env: WorldProxyEnv,
    block: LogBlock,
    story: list[str],
    timeline: list[str],
) -> bool:
    parsed_output = _safe_parse_output(block.llm_output)
    if not isinstance(parsed_output, dict):
        return False

    new_goal = str(parsed_output.get("goal", "")).strip()
    if not new_goal:
        return False

    previous_goal_match = re.search(r"^Previous goal:\s*(.+)$", block.user_prompt, flags=re.MULTILINE)
    previous_goal = previous_goal_match.group(1).strip() if previous_goal_match else env.get_current_goal()
    world_before = env.get_world_vars()
    previous_goal_domain = str(world_before.get("current_goal_domain", "")).strip()
    goal_domain = str(parsed_output.get("goal_domain", "")).strip()
    rationale = str(parsed_output.get("rationale", "")).strip()
    transition_paragraph = str(parsed_output.get("transition_paragraph", "")).strip()

    env.apply_world_updates(
        {
            "previous_goal": previous_goal,
            "previous_goal_domain": previous_goal_domain,
            "current_goal_domain": goal_domain,
            "goal_shift_rationale": rationale,
        }
    )
    env.set_new_goal(new_goal)
    if transition_paragraph:
        story.append(transition_paragraph)
        timeline.append(f"recovered_t={env.get_world_vars().get('turn', 0)} transition_story={transition_paragraph}")
    timeline.append(f"recovered_new_goal previous_goal={previous_goal} new_goal={new_goal}")
    return True


def _extract_json_field(user_prompt: str, field_name: str) -> Any:
    marker = f"{field_name}:"
    start = user_prompt.find(marker)
    if start == -1:
        return None
    start += len(marker)
    decoder = json.JSONDecoder()
    tail = user_prompt[start:].lstrip()
    try:
        value, _ = decoder.raw_decode(tail)
    except json.JSONDecodeError:
        return None
    return value


def _safe_parse_output(content: str) -> dict[str, Any] | None:
    try:
        parsed = _parse_json_object_with_repair(content, "LLM log recovery")
    except LLMError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _merged_story_prefix(existing_story: list[str], user_prompt: str) -> list[str]:
    recent_story = _extract_json_field(user_prompt, "Recent story paragraphs")
    if not isinstance(recent_story, list):
        return list(existing_story)

    cleaned_recent = [str(item).strip() for item in recent_story if str(item).strip()]
    if not cleaned_recent:
        return []
    if not existing_story:
        return cleaned_recent

    prefix = list(existing_story)
    overlap = min(len(prefix), len(cleaned_recent))
    for size in range(overlap, 0, -1):
        if prefix[-size:] == cleaned_recent[:size]:
            return prefix + cleaned_recent[size:]
    return cleaned_recent


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
