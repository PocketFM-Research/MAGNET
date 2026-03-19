from __future__ import annotations
import json
from typing import Any

FEW_SHOT_ACTION_EXAMPLES = """
Example 1
Character persona: cautious medic trying to evacuate civilians.
World variables: {"alarm_active": true, "exit_blocked": false, "goal_reached": false}
Goal: move injured civilians toward safety without causing panic.
Output JSON: {"action": "guide the two most injured civilians through the east corridor to the marked shelter door", "confidence": 0.9, "rationale": "It is concrete, immediately reduces danger, and creates measurable progress toward full evacuation."}

Example 2
Character persona: ambitious engineer under tight deadline pressure.
World variables: {"prototype_failed": true, "time_remaining_hours": 3, "goal_reached": false}
Goal: recover from failure and deliver a workable demo.
Output JSON: {"action": "replace the unstable sensor module with the tested backup and rerun the core demo sequence", "confidence": 0.92, "rationale": "This directly addresses the failure source and advances the story with a high-impact recovery step instead of stalling."}

Example 3
Character persona: community organizer who values trust and accountability.
World variables: {"team_conflict_open": true, "resources_secured": false, "goal_reached": false}
Goal: resolve internal conflict so the team can secure supplies.
Output JSON: {"action": "hold a 10-minute mediation between the two lead volunteers and assign clear pickup roles before departure", "confidence": 0.89, "rationale": "It resolves a blocking conflict and enables immediate next actions tied to the final objective."}
""".strip()
def build_action_prompt(
    name: str,
    persona: str,
    goal: str,
    world_vars: dict[str, Any],
    memory_snippets: list[str],
    world_knowledge: list[str],
    last_scene_summary: str,
    previous_scene_summary: str,
    revision_feedback: str | None,
) -> tuple[str, str]:
    system = (
        f"You are the action generator for {name}. "
        f"Character persona: {persona} "
        "Output one concrete next action for the character. "
        "Be concise, realistic, and consistent with world variables, persona, and the current story goal."
    )
    feedback_line = f"Revision feedback: {revision_feedback}\n" if revision_feedback else ""
    user = (
        f"TASK=action\n"
        f"Character: {name}\n"
        f"Goal: {goal}\n"
        # f"World variables: {json.dumps(world_vars, sort_keys=True)}\n"
        # f"Retrieved memory: {json.dumps(memory_snippets)}\n" 
        "World knowledge:\n"
        f"You know: {json.dumps(world_knowledge)}\n"
        "\n"
        "Recent history:\n"
        f"Last scene: {last_scene_summary}\n"
        f"Scene before that: {previous_scene_summary}\n\n"
        f"{feedback_line}"
        # f"Few-shot examples:\n{FEW_SHOT_ACTION_EXAMPLES}\n"
        "\n"
        "Return JSON keys: action (string), confidence (0..1), rationale (string)."
    )
    return system, user


def build_critic_prompt(
    name: str,
    persona: str,
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
        f"Character persona: {persona}\n"
        f"Action: {action}\n"
        f"Story goal: {goal}\n"
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
    proposals: list[dict[str, Any]],
    world_before: dict[str, Any],
) -> tuple[str, str]:
    system = (
        "You are a story narrator deciding which proposed character actions become canonical story events for this timestep. "
        "Select only the actions that materially belong in the story beat and progress the story, then narrate just those chosen events to progress the story without repeating previous events. "
        "Maintain continuity of cause-and-effect and character motivations. "
        "Never mention acts, phases, scene numbers, simulation mechanics, or timeline labels."
    )
    user = (
        "TASK=narrate_step\n"
        f"Story goal: {story_goal}\n"
        f"Recent story paragraphs: {json.dumps(recent_story)}\n"
        f"Proposed actions: {json.dumps(proposals, sort_keys=True)}\n"
        f"World before: {json.dumps(world_before, sort_keys=True)}\n"
        "Write one paragraph (2-4 sentences) in plain past-tense prose. "
        "Do not use phrases like 'Act 1, Act 2, Act 3, ...', 'phase', 'stage', or 'timeline'. "
        "Do not add facts that conflict with the selected proposals or world vars. "
        "Choose a small subset of proposals that best advances or meaningfully develops the current story beat; it is normal to omit many proposals. "
        "Prefer 1-2 selected actions unless multiple actions are tightly linked by cause and effect. "
        "If one selected action reaches the goal, do not select later unrelated actions in the same timestep. "
        "Return JSON keys: included_indices (list[int]), paragraph (string), continuity_note (string)."
    )
    return system, user


def build_new_goal_prompt(
    completed_goal: str,
    recent_story: list[str],
    world_vars: dict[str, Any],
    character_context: list[dict[str, Any]],
    goal_history: list[str],
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
        f"Available characters: {json.dumps(character_context, sort_keys=True)}\n"
        f"Goal history: {json.dumps(goal_history)}\n"
        "Write a new story goal that follows from the completed one, raises or redirects the stakes, "
        "does not repeat past goals, and remains achievable through future character actions. "
        "You may shift attention to an available character who was not central to the last goal, "
        "but the new goal must still grow out of the recent story and current world state. "
        "When introducing a less-used character, make that character's motive or ability relevant to the next conflict. "
        "Avoid repeating the completed goal verbatim or ending the story. "
        "Return JSON keys: goal (string), rationale (string)."
    )
    return system, user
