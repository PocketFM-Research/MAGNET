from __future__ import annotations
import json
from typing import Any

FEW_SHOT_ACTION_EXAMPLES = """
Example 1
Character persona: ant focused on gathering food.
World variables: {"phase": "acts", "current_act": 1, "ant_in_water": false, "hunter_present": false}
Intent: advance setup toward the river accident.
Output JSON: {"action": "forage near river", "confidence": 0.88, "rationale": "This sets up the Act 1 accident."}

Example 2
Character persona: dove who helps nearby creatures.
World variables: {"phase": "acts", "current_act": 2, "ant_in_water": true, "hunter_present": false}
Intent: rescue the ant.
Output JSON: {"action": "rescue ant with leaf", "confidence": 0.93, "rationale": "Act 2 requires immediate rescue."}

Example 3
Character persona: ant who remembers being saved.
World variables: {"phase": "acts", "current_act": 4, "dove_endangered": true, "hunter_present": true}
Intent: resolve danger by helping the dove survive.
Output JSON: {"action": "bite hunter leg", "confidence": 0.95, "rationale": "Act 4 mirrors the original fable payoff."}
""".strip()


def build_intent_prompt(
    name: str,
    persona: str,
    goal: str,
    world_vars: dict[str, Any],
    memory_snippets: list[str],
) -> tuple[str, str]:
    system = (
        "You are the intent planner for a character in a text simulation. "
        "Use persona + current world state to produce the next high-level intent."
    )
    user = (
        f"TASK=intent\n"
        f"Character: {name}\n"
        f"Persona: {persona}\n"
        f"Goal: {goal}\n"
        f"World variables: {json.dumps(world_vars, sort_keys=True)}\n"
        f"Retrieved memory: {json.dumps(memory_snippets)}\n"
        "Return JSON keys: intent (string), confidence (0..1), constraints (list[str])."
    )
    return system, user


def build_action_prompt(
    name: str,
    persona: str,
    intent: str,
    constraints: list[str],
    world_vars: dict[str, Any],
    memory_snippets: list[str],
    revision_feedback: str | None,
) -> tuple[str, str]:
    system = (
        "You are the action generator. Output one concrete next action for the character. "
        "Be concise, realistic, and consistent with world variables and current narrative act."
    )
    feedback_line = f"Revision feedback: {revision_feedback}\n" if revision_feedback else ""
    user = (
        f"TASK=action\n"
        f"Character: {name}\n"
        f"Persona: {persona}\n"
        f"Intent: {intent}\n"
        f"Constraints: {json.dumps(constraints)}\n"
        f"World variables: {json.dumps(world_vars, sort_keys=True)}\n"
        f"Retrieved memory: {json.dumps(memory_snippets)}\n"
        f"{feedback_line}"
        f"Few-shot examples:\n{FEW_SHOT_ACTION_EXAMPLES}\n"
        "Return JSON keys: action (string), confidence (0..1), rationale (string)."
    )
    return system, user


def build_critic_prompt(
    name: str,
    action: str,
    world_vars: dict[str, Any],
) -> tuple[str, str]:
    system = (
        "You are an action critic. Decide whether the proposed action should be revised. "
        "Prefer revisions when the action conflicts with required progression of the current act."
    )
    user = (
        f"TASK=critic\n"
        f"Character: {name}\n"
        f"Action: {action}\n"
        f"World variables: {json.dumps(world_vars, sort_keys=True)}\n"
        "Return JSON keys: revise (boolean), confidence (0..1), feedback (string)."
    )
    return system, user


def build_narrator_prompt(
    story_goal: str,
    opening: str,
    recent_story: list[str],
    actor: str,
    intent: str,
    action: str,
    event_text: str,
    world_before: dict[str, Any],
    world_after: dict[str, Any],
) -> tuple[str, str]:
    system = (
        "You are a story narrator. Rewrite simulation events into a cohesive story while staying faithful to facts. "
        "Maintain continuity of cause-and-effect and character motivations. "
        "Never mention acts, phases, scene numbers, simulation mechanics, or timeline labels."
    )
    user = (
        "TASK=narrate_step\n"
        f"Story goal: {story_goal}\n"
        f"Opening: {opening}\n"
        f"Recent story paragraphs: {json.dumps(recent_story)}\n"
        f"Actor: {actor}\n"
        f"Intent: {intent}\n"
        f"Action: {action}\n"
        f"Event text: {event_text}\n"
        f"World before: {json.dumps(world_before, sort_keys=True)}\n"
        f"World after: {json.dumps(world_after, sort_keys=True)}\n"
        "Write one paragraph (2-4 sentences) in plain past-tense prose. "
        "Do not use phrases like 'Act 1, Act 2, Act 3, ...', 'phase', 'stage', or 'timeline'. "
        "Do not add facts that conflict with event text/world vars. "
        "Focus on character motivations, emotional tone, and relationships. "
        "Explain why the character took the action and how it affects others. "
        "Use personality traits and goals when describing actions. "
        "Return JSON keys: paragraph (string), continuity_note (string)."
    )
    return system, user
