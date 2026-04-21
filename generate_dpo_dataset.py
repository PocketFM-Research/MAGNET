from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from llm import LLMError, build_default_llm
from prompts import build_action_prompt, build_critic_prompt, build_new_goal_prompt
from sim_types import CharacterDecision, CharacterProfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DPO preference data from the current character action pipeline."
    )
    parser.add_argument("--fable", default="maya story", help="Fable/story definition name.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of rollout episodes.")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum steps per episode.")
    parser.add_argument(
        "--output",
        default="artifacts/dpo_preferences.jsonl",
        help="Path to the output JSONL dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file instead of appending to it.",
    )
    parser.add_argument(
        "--rag-k",
        type=int,
        default=0,
        help="Number of memory snippets retrieved for each character decision.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature used for candidate actions.",
    )
    parser.add_argument(
        "--max-new-goals",
        type=int,
        default=1,
        help="Maximum number of follow-up goals to generate per episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic candidate ordering and retry variation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    from agents import NarratorAgent
    from environment import WorldProxyEnv
    from fables import get_fable_definition

    llm = build_default_llm()
    fable = get_fable_definition(args.fable)
    env = WorldProxyEnv(fable=fable)
    narrator = NarratorAgent(llm=llm)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    output_mode = "w" if args.overwrite else "a"
    with output_path.open(output_mode, encoding="utf-8") as handle:
        for episode_index in range(args.episodes):
            rows_written += run_episode(
                llm=llm,
                env=env,
                narrator=narrator,
                characters=fable.characters,
                output_handle=handle,
                episode_index=episode_index,
                max_steps=args.max_steps,
                rag_k=args.rag_k,
                temperature=args.temperature,
                max_new_goals=args.max_new_goals,
            )

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "rows_written": rows_written,
                "episodes": args.episodes,
                "fable": fable.name,
            },
            indent=2,
        )
    )


def run_episode(
    llm: Any,
    env: Any,
    narrator: Any,
    characters: list[CharacterProfile],
    output_handle: Any,
    episode_index: int,
    max_steps: int,
    rag_k: int,
    temperature: float,
    max_new_goals: int,
) -> int:
    env.reset([profile.name for profile in characters])
    story: list[str] = []
    memory = build_memory() if rag_k > 0 else None
    rows_written = 0
    new_goals_used = 0
    goal_assigned_step = 1

    for step in range(1, max_steps + 1):
        world_before = env.get_world_vars()
        active_goal = env.get_current_goal()
        proposed_actions: list[CharacterDecision] = []

        for profile in characters:
            memory_snippets = (
                memory.retrieve(
                    profile.name,
                    query=f"goal={active_goal} world={world_before}",
                    k=rag_k,
                )
                if memory is not None
                else []
            )
            prompt_payload = build_prompt_payload(
                profile=profile,
                goal=active_goal,
                world_vars=world_before,
                memory_snippets=memory_snippets,
                recent_story=story[-4:],
            )

            candidates = sample_candidates(
                llm=llm,
                prompt_payload=prompt_payload,
                temperature=temperature,
            )
            if len(candidates) < 2:
                continue

            preference = judge_candidates(
                llm=llm,
                profile=profile,
                goal=active_goal,
                world_vars=world_before,
                recent_story=story[-4:],
                candidates=candidates,
            )
            chosen_idx = preference["chosen_index"]
            rejected_idx = 1 - chosen_idx
            chosen_action = candidates[chosen_idx]
            rejected_action = candidates[rejected_idx]
            chosen_eval = critique_action(
                llm=llm,
                profile=profile,
                action=chosen_action["action"],
                goal=active_goal,
                world_vars=world_before,
            )

            output_handle.write(
                json.dumps(
                    {
                        "episode": episode_index,
                        "step": step,
                        "character": profile.name,
                        "prompt": prompt_payload,
                        "chosen": chosen_action["action"],
                        "rejected": rejected_action["action"],
                        "chosen_rationale": chosen_action["rationale"],
                        "rejected_rationale": rejected_action["rationale"],
                        "judge": preference,
                        "chosen_eval": chosen_eval,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            rows_written += 1

            proposed_actions.append(
                CharacterDecision(
                    character=profile.name,
                    action=chosen_action["action"],
                    confidence=float(chosen_action["confidence"]),
                    revisions_used=0,
                    rationale=chosen_action["rationale"],
                    advances_goal=bool(chosen_eval.get("advances_goal", False)),
                    goal_reached=bool(chosen_eval.get("goal_reached", False)),
                    world_updates=chosen_eval.get("world_updates", {})
                    if isinstance(chosen_eval.get("world_updates", {}), dict)
                    else {},
                    progress_reason=str(chosen_eval.get("reason", "")),
                )
            )

        if not proposed_actions:
            if step - goal_assigned_step + 1 >= 15:
                goal_assigned_step = refresh_stale_goal(
                    llm=llm,
                    env=env,
                    characters=characters,
                    story=story,
                    step=step,
                    active_goal=active_goal,
                    world_after=world_before,
                )
            continue

        narrated_step = narrator.narrate_step(
            story_goal=active_goal,
            recent_story=story[-3:],
            world_before=world_before,
            proposals=proposed_actions,
        )
        selected_actions = choose_selected_actions(proposed_actions, narrated_step.included_indices)
        if not selected_actions:
            if step - goal_assigned_step + 1 >= 15:
                goal_assigned_step = refresh_stale_goal(
                    llm=llm,
                    env=env,
                    characters=characters,
                    story=story,
                    step=step,
                    active_goal=active_goal,
                    world_after=world_before,
                )
            continue

        results = env.step_selected_actions(selected_actions)
        world_after = env.get_world_vars()
        step_reward = sum(result.reward for result in results)
        story.append(narrated_step.paragraph)

        if memory is not None:
            memory.add(
                timestep=step,
                characters=[decision.character for decision in selected_actions],
                actions=[decision.action for decision in selected_actions],
                narration=narrated_step.paragraph,
                reward=step_reward,
                world_before=world_before,
                world_after=world_after,
            )

        completed_result = next((result for result in results if result.info.get("goal_completed")), None)
        if completed_result is not None and new_goals_used < max_new_goals:
            new_goal = generate_next_goal(
                llm=llm,
                narrator=narrator,
                completed_goal=str(completed_result.info.get("completed_goal", active_goal)),
                story=story,
                world_after=world_after,
                characters=characters,
            )
            env.set_new_goal(new_goal)
            new_goals_used += 1
            goal_assigned_step = step + 1
        elif step - goal_assigned_step + 1 >= 15:
            goal_assigned_step = refresh_stale_goal(
                llm=llm,
                env=env,
                characters=characters,
                story=story,
                step=step,
                active_goal=active_goal,
                world_after=world_after,
            )

    return rows_written


def build_memory() -> Any:
    from memory import StructuredMemory

    try:
        return StructuredMemory()
    except Exception:
        return None


def build_prompt_payload(
    profile: CharacterProfile,
    goal: str,
    world_vars: dict[str, Any],
    memory_snippets: list[str],
    recent_story: list[str],
) -> dict[str, Any]:
    world_knowledge = build_world_knowledge(profile, world_vars)
    recent_scene_summaries = recent_story or ["No prior scene has been narrated yet."]
    return {
        "character": profile.name,
        "goal": goal,
        "persona": profile.persona_text(),
        "world_vars": world_vars,
        "memory_snippets": memory_snippets,
        "world_knowledge": world_knowledge,
        "recent_story": recent_scene_summaries,
    }


def sample_candidates(
    llm: Any,
    prompt_payload: dict[str, Any],
    temperature: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    used_actions: set[str] = set()

    for attempt in range(6):
        variant_tag = f"variant-{attempt + 1}"
        system_prompt, user_prompt = build_action_prompt(
            prompt_payload["character"],
            prompt_payload["persona"],
            prompt_payload["goal"],
            prompt_payload["world_vars"],
            prompt_payload["memory_snippets"],
            prompt_payload["world_knowledge"],
            prompt_payload["recent_story"],
            (
                f"Produce a distinct option labeled {variant_tag}. Avoid repeating prior variants."
                if candidates
                else None
            ),
        )
        try:
            response = llm.complete_json(system_prompt, user_prompt, temperature=temperature)
        except LLMError:
            continue
        action = str(response.get("action", "")).strip()
        if not action:
            continue
        normalized = action.casefold()
        if normalized in used_actions:
            continue
        used_actions.add(normalized)
        candidates.append(
            {
                "action": action,
                "confidence": normalize_confidence(response.get("confidence", 0.5)),
                "rationale": str(response.get("rationale", "")).strip(),
            }
        )
        if len(candidates) == 2:
            break

    return candidates


def critique_action(
    llm: Any,
    profile: CharacterProfile,
    action: str,
    goal: str,
    world_vars: dict[str, Any],
) -> dict[str, Any]:
    critic_sys, critic_user = build_critic_prompt(
        profile.name,
        profile.persona_text(),
        action,
        goal,
        world_vars,
    )
    return llm.complete_json(critic_sys, critic_user)


def judge_candidates(
    llm: Any,
    profile: CharacterProfile,
    goal: str,
    world_vars: dict[str, Any],
    recent_story: list[str],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    system_prompt = (
        "You are a strict pairwise critic for character actions. "
        "Choose the better next action for the character. "
        "Prefer actions that are concrete, less repetitive, more in-character, "
        "and more likely to push the story forward through visible consequence or interaction. "
        "Penalize filler, hesitation with no consequence, generic emotional restatement, "
        "and actions that merely repeat the recent beat."
    )
    user_prompt = (
        "TASK=pairwise_action_judge\n"
        f"Character: {profile.name}\n"
        f"Persona: {profile.persona_text()}\n"
        f"Goal: {goal}\n"
        f"World variables: {json.dumps(world_vars, sort_keys=True)}\n"
        f"Recent story: {json.dumps(recent_story)}\n"
        f"Candidate 0: {json.dumps(candidates[0], sort_keys=True)}\n"
        f"Candidate 1: {json.dumps(candidates[1], sort_keys=True)}\n"
        "Return JSON with keys: chosen_index (0 or 1), reason (string), "
        "goal_progress (object with keys 0 and 1), repetition_risk (object with keys 0 and 1), "
        "specificity (object with keys 0 and 1), character_fit (object with keys 0 and 1)."
    )
    preference = llm.complete_json(system_prompt, user_prompt)
    chosen_index = preference.get("chosen_index", 0)
    if chosen_index not in (0, 1):
        preference["chosen_index"] = 0
    return preference


def choose_selected_actions(
    proposed_actions: list[CharacterDecision],
    included_indices: list[int],
) -> list[CharacterDecision]:
    selected: list[CharacterDecision] = []
    seen_indices: set[int] = set()
    for idx in included_indices:
        if not isinstance(idx, int):
            continue
        if idx < 0 or idx >= len(proposed_actions):
            continue
        if idx in seen_indices:
            continue
        seen_indices.add(idx)
        selected.append(proposed_actions[idx])
        if proposed_actions[idx].goal_reached:
            break

    if selected:
        return selected
    return proposed_actions[:1]


def generate_next_goal(
    llm: Any,
    narrator: Any,
    completed_goal: str,
    story: list[str],
    world_after: dict[str, Any],
    characters: list[CharacterProfile],
) -> str:
    goal_sys, goal_user = build_new_goal_prompt(
        completed_goal=completed_goal,
        recent_story=story[-3:],
        world_vars=world_after,
        character_context=build_character_context(characters),
        goal_history=list(world_after.get("goal_history", [completed_goal])),
        goal_status="completed",
    )
    try:
        goal_resp = llm.complete_json(goal_sys, goal_user)
    except LLMError:
        return completed_goal

    next_goal = str(goal_resp.get("goal", "")).strip()
    return next_goal or completed_goal


def refresh_stale_goal(
    llm: Any,
    env: Any,
    characters: list[CharacterProfile],
    story: list[str],
    step: int,
    active_goal: str,
    world_after: dict[str, Any],
) -> int:
    goal_sys, goal_user = build_new_goal_prompt(
        completed_goal=active_goal,
        recent_story=story[-3:],
        world_vars=world_after,
        character_context=build_character_context(characters),
        goal_history=list(world_after.get("goal_history", [active_goal])),
        goal_status="not achieved after 15 steps since assignment; replace it with a feasible goal for the current story state",
    )
    try:
        goal_resp = llm.complete_json(goal_sys, goal_user)
    except LLMError:
        return step - 14

    new_goal = str(goal_resp.get("goal", "")).strip()
    if not new_goal or new_goal == active_goal:
        return step - 14

    env.set_new_goal(new_goal)
    return step + 1


def build_character_context(characters: list[CharacterProfile]) -> list[dict[str, Any]]:
    return [
        {
            "name": profile.name,
            "role": profile.role,
            "traits": profile.traits,
            "goals": profile.goals,
            "abilities": profile.abilities,
            "relationships": profile.relationships,
            "state": profile.state,
        }
        for profile in characters
    ]


def build_world_knowledge(profile: CharacterProfile, world_vars: dict[str, Any]) -> list[str]:
    facts: list[str] = []
    seen_keys: set[str] = set()
    ignored_keys = {
        "characters",
        "current_goal",
        "fable_name",
        "goal_history",
        "goal_reached",
        "progress_count",
        "turn",
    }

    lowered_name = profile.name.lower()
    for key in sorted(world_vars):
        if key in seen_keys or len(facts) >= 5 or key in ignored_keys:
            continue
        value = world_vars[key]
        value_text = str(value).lower()
        key_text = key.lower()
        if lowered_name in key_text or lowered_name in value_text:
            facts.append(f"{key}={world_vars[key]}")
            seen_keys.add(key)

    for key in sorted(world_vars):
        if key in seen_keys or len(facts) >= 5 or key in ignored_keys:
            continue
        value = world_vars[key]
        key_text = key.lower()
        value_text = str(value).lower()
        if any(token in key_text for token in ("action", "event", "actor", "status", "state")):
            facts.append(f"{key}={world_vars[key]}")
            seen_keys.add(key)
            continue
        if any(name.lower() in value_text for name in profile.relationships) or lowered_name in value_text:
            facts.append(f"{key}={world_vars[key]}")
            seen_keys.add(key)

    for key in sorted(world_vars):
        if key in seen_keys or len(facts) >= 5 or key in ignored_keys:
            continue
        value = world_vars[key]
        if isinstance(value, str) and value.strip():
            facts.append(f"{key}={value}")
            seen_keys.add(key)

    return facts[:5] or ["No relevant character events are currently known."]


def normalize_confidence(value: Any) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return 0.5
        return max(0.0, min(1.0, parsed))
    return 0.5


if __name__ == "__main__":
    main()
