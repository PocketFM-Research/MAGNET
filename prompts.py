from __future__ import annotations
import json
from typing import Any

FEW_SHOT_ACTION_EXAMPLES = """
Example 1
Character persona: cautious medic trying to evacuate civilians.
World variables: {"alarm_active": true, "exit_blocked": false, "goal_reached": false}
Intent: move injured civilians toward safety without causing panic.
Output JSON: {"action": "guide the two most injured civilians through the east corridor to the marked shelter door", "confidence": 0.9, "rationale": "It is concrete, immediately reduces danger, and creates measurable progress toward full evacuation."}

Example 2
Character persona: ambitious engineer under tight deadline pressure.
World variables: {"prototype_failed": true, "time_remaining_hours": 3, "goal_reached": false}
Intent: recover from failure and deliver a workable demo.
Output JSON: {"action": "replace the unstable sensor module with the tested backup and rerun the core demo sequence", "confidence": 0.92, "rationale": "This directly addresses the failure source and advances the story with a high-impact recovery step instead of stalling."}

Example 3
Character persona: community organizer who values trust and accountability.
World variables: {"team_conflict_open": true, "resources_secured": false, "goal_reached": false}
Intent: resolve internal conflict so the team can secure supplies.
Output JSON: {"action": "hold a 10-minute mediation between the two lead volunteers and assign clear pickup roles before departure", "confidence": 0.89, "rationale": "It resolves a blocking conflict and enables immediate next actions tied to the final objective."}
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
        "Use persona, goal, and current world state to produce the next high-level intent."
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
        "Be concise, realistic, and consistent with world variables, persona, and the final story goal."
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
    goal: str,
    world_vars: dict[str, Any],
) -> tuple[str, str]:
    system = (
        "You are a strict action critic. Decide whether the proposed action should be revised. "
        "Evaluate both action quality and whether the action plausibly advances the current story goal. "
        "Focus on whether the action is specific, non-redundant with previous actions, plausible in the current world, "
        "consistent with the character's persona, and materially relevant to the current goal. "
        "Revise when the action is vague, repetitive, out of character, implausible, conflicts with world state, or does not advance the goal. "
        "Be conservative: approve only actions that are concrete, believable, narratively coherent, and goal-relevant."
    )
    user = (
        f"TASK=critic\n"
        f"Character: {name}\n"
        f"Action: {action}\n"
        f"Final goal: {goal}\n"
        f"World variables: {json.dumps(world_vars, sort_keys=True)}\n"
        "Reject actions that contradict persona, violate the established world, repeat recent state without adding anything new, "
        "read as implausible filler, or fail to create credible goal progress.\n"
        "If advances_goal=true and goal_reached=false, world_updates must include at least one concrete state change. "
        "Do not write to reserved keys: turn, characters, fable_name, current_goal, goal_history.\n"
        "Use feedback to demand a more concrete, less repetitive, more believable, more goal-relevant next action.\n"
        "Return JSON keys: revise (boolean), advances_goal (boolean), goal_reached (boolean), "
        "world_updates (object), confidence (0..1), feedback (string), reason (string)."
    )
    return system, user


def build_narrator_prompt(
    story_goal: str,
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


def build_new_goal_prompt(
    completed_goal: str,
    recent_story: list[str],
    world_vars: dict[str, Any],
) -> tuple[str, str]:
    system = (
        "You are a story narrator extending the story after a goal has just been achieved. "
        "Create the next concrete story goal so the narrative continues naturally from the current state."
    )
    user = (
        "TASK=new_goal\n"
        f"Completed goal: {completed_goal}\n"
        f"Recent story paragraphs: {json.dumps(recent_story)}\n"
        f"World variables: {json.dumps(world_vars, sort_keys=True)}\n"
        "Write a new story goal that follows from the completed one, raises or redirects the stakes, "
        "does not repeat past goals, and remains achievable through future character actions. "
        "Avoid repeating the completed goal verbatim or ending the story. "
        "Return JSON keys: goal (string), rationale (string)."
    )
    return system, user
