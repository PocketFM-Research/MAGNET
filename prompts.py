from __future__ import annotations
import json
from typing import Any

def build_action_prompt(
    name: str,
    persona: str,
    goal: str,
    world_vars: dict[str, Any],
    memory_snippets: list[str],
    world_knowledge: list[str],
    recent_scene_summaries: list[str],
    revision_feedback: str | None,
) -> tuple[str, str]:
    system = (
        f"You are the action generator for {name}. "
        f"Character persona: {persona} "
        "Output one concrete next action for the character, but choose it from inside the character's present interpretation of the moment. "
        "Be concise, realistic, psychologically specific, and consistent with world variables, persona, and the current story goal."
    )
    feedback_line = f"Revision feedback: {revision_feedback}\n" if revision_feedback else ""
    history_lines = "\n".join(
        f"Recent scene {idx + 1}: {summary}"
        for idx, summary in enumerate(recent_scene_summaries)
    )
    user = (
        f"TASK=action\n"
        f"Character: {name}\n"
        f"Goal: {goal}\n"
        # f"World variables: {json.dumps(world_vars, sort_keys=True)}\n"
        # f"Retrieved memory: {json.dumps(memory_snippets)}\n" 
        "World knowledge:\n"
        f"You know: {json.dumps(world_knowledge)}\n"
        f"Retrieved emotional memory: {json.dumps(memory_snippets)}\n"
        "\n"
        "Recent history:\n"
        f"{history_lines}\n\n"
        f"{feedback_line}"
        "\n"
        "Choose an action that reveals how the character is interpreting the situation while still pushing toward goal completion. "
        "It is valid for emotions like fear, loyalty, guilt, resentment, grief, insecurity, desire for control, and remembered prior scenes shape a character's action. "
        "It is valid for the character to pause, hesitate, deflect, misread another person, protect a secret, ask the wrong question, soften, harden, or choose a smaller practical move because a direct one costs too much emotionally. "
        "Do not add melodrama, speeches about feelings, or therapy-like self-analysis; keep the action observable and grounded.\n"
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
        "Evaluate both action quality and whether the action plausibly advances the current story goal or deepens the emotional situation that makes the goal matter. "
        "Focus on whether the action is specific, non-redundant with previous actions, plausible in the current world, "
        "consistent with the character's persona, and materially relevant to the current goal, relationship pressure, or internal contradiction. "
        "Do not require every good action to be efficient, cooperative, escalatory, or revealing. "
        "Revise when the action is vague, repetitive, out of character, implausible, conflicts with world state, or adds neither plot consequence nor emotional/perceptual consequence. "
        "Be conservative: approve only actions that are concrete, believable, narratively coherent, and meaningful."
    )
    user = (
        f"TASK=critic\n"
        f"Character: {name}\n"
        f"Character persona: {persona}\n"
        f"Action: {action}\n"
        f"Story goal: {goal}\n"
        f"World variables: {json.dumps(world_vars, sort_keys=True)}\n"
        "Reject actions that contradict persona, violate the established world, repeat recent state without adding anything new, "
        "read as implausible filler, or fail to create credible plot, relationship, emotional, or perceptual progress.\n"
        "Approve emotionally meaningful restraint, hesitation, deflection, misinterpretation, or partial disclosure when it changes what a character risks, believes, withholds, or perceives. "
        "If advances_goal=true and goal_reached=false, world_updates must include at least one concrete state change, which may be emotional, relational, perceptual, or practical. "
        "Do not write to reserved keys: turn, characters, fable_name, current_goal, goal_history.\n"
        "Use feedback to demand a more concrete, less repetitive, more believable, more psychologically specific next action.\n"
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
    opening_scene = not recent_story
    system = (
        "You are a story narrator deciding which proposed character actions become canonical story events for this timestep. "
        "Select only the actions that materially belong in the story beat and turn them into a coherent scene, not a stitched summary of moves. "
        "Maintain continuity of cause-and-effect, spatial logic, character motivations, and the physical setting. "
        "Treat emotional interpretation, relationship shifts, subtext, hesitation, avoidance, and changed perception as real story events, not just logistical or tactical progress. "
        "Use the world state to keep the environment vivid and specific when it matters, including believable setting pressure, movement, obstacles, or changes in conditions. "
        "Let objects and settings carry memory or emotional association when supported by recent story, but keep this restrained and concrete. "
        "Prefer plain prose over lyrical phrasing; use figurative language rarely and do not repeatedly personify objects, rooms, silence, weather, or documents. "
        "You may add small connective narration or grounded setting detail if it is strongly supported by the selected proposals and world vars, but do not invent major new events. "
        "If the story clearly depends on a missing concrete detail that has never been stated, you may supply that detail yourself as long as it is plausible, specific, and consistent with the selected proposals, recent story, and world vars. "
        "Use this sparingly to fill narrative gaps such as the contents of documents, the exact accusation being made, or another essential fact the scene cannot fully land without; prefer the smallest story-useful addition rather than a twist. "
        "Never mention acts, phases, scene numbers, simulation mechanics, or timeline labels."
    )
    opening_guidance = (
        "This is the opening scene. Start smoothly and naturally, like the beginning of a real novel chapter rather than a recap. "
        "You may spend the first 1-2 sentences quietly establishing the setting, immediate pressure, and the relevant character situation before landing the chosen actions. "
        "If helpful, briefly frame who a character is in the moment through behavior or role, but do it in flowing prose rather than exposition dumps. "
        "Favor clarity, atmosphere, and narrative momentum over compression. "
    ) if opening_scene else ""
    user = (
        "TASK=narrate_step\n"
        f"Story goal: {story_goal}\n"
        f"Recent story paragraphs: {json.dumps(recent_story)}\n"
        f"Proposed actions: {json.dumps(proposals, sort_keys=True)}\n"
        f"World before: {json.dumps(world_before, sort_keys=True)}\n"
        f"{opening_guidance}"
        f"Write one paragraph ({'4-6' if opening_scene else '3-5'} sentences) in plain past-tense prose. "
        "The paragraph should feel like a real scene unfolding beat by beat, with actions influencing one another inside a clear setting and changing what the characters believe, fear, expect, or can bear to say. "
        "It is fine to briefly mention environmental developments, pressure, or visible consequences if they naturally follow from the selected actions and existing world vars. "
        "Let emotional reinterpretation, relational tension, and subtext matter when they are present, but keep the visible action of the scene clear. "
        "Allow pauses, silence, implication, avoidance, and partial understanding; not every paragraph needs escalation or efficient goal completion. "
        "Keep the prose mostly literal and observational, and favor exact sensory or social detail over decorative language. "
        "Only use a metaphor, simile, or symbolic echo when strongly earned by the scene. "
        "Do not explain feelings in abstract labels or therapy-style language. Instead, reveal inner change through concrete perception, remembered detail, word choice, gesture, and what remains unsaid. "
        "If an important story fact is missing but strongly implied, you may state a plausible version of it so the scene can resolve clearly, provided it does not conflict with prior material. "
        "Do not merely list what each character did one after another. Build a natural flow with transitions, reactions, and consequences. "
        "Do not use phrases like 'Act 1, Act 2, Act 3, ...', 'phase', 'stage', or 'timeline'. "
        "Do not add facts that conflict with the selected proposals or world vars. "
        "Do not end the paragraph with a teaser or question for the next scene."
        "Choose a small subset of proposals that best advances or meaningfully develops the current story beat, including emotional or relational development; it is normal to omit many proposals. "
        "Prefer 1-2 selected actions unless multiple actions are tightly linked by cause and effect or are needed to make the scene readable. "
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
    goal_status: str = "completed",
    force_domain_shift: bool = False,
    previous_goal_domain: str = "",
) -> tuple[str, str]:
    system = (
        "You are a story narrator extending or redirecting the story after a goal changes. "
        "Create the next concrete story goal so the narrative continues naturally from the current state."
    )
    domain_shift_lines = ""
    transition_lines = (
        "Write a short transition paragraph in plain past-tense prose that closes the previous goal through already-supported consequences and creates the opening pressure of the new goal. "
        "This paragraph should feel like real story, not summary notes or metadata. "
        "Do not introduce a major new event; use it to bridge what just happened into what now matters. "
    )
    return_keys = "transition_paragraph (string), goal_domain (string), goal (string), rationale (string)"
    if force_domain_shift:
        transition_lines = (
            "Write a short transition paragraph in plain past-tense prose that closes the previous conflict and creates the opening pressure of that specific new goal in the new domain. "
            "This paragraph should feel like real story, not summary notes or metadata, and may make the emotional turn into the new domain visible through concrete setting, gesture, memory, or subtext. "
            "Do not introduce a major new event; use it to bridge what just happened into what now matters. "
        )
        domain_shift_lines = (
            f"Previous goal domain: {previous_goal_domain or 'unknown'}\n"
            "Force domain shift: true\n"
            "The new goal must belong to a different life domain than the previous goal. "
            "Do not merely escalate the same investigation, argument, or procedural struggle. "
            "Prefer major shifts such as legal to domestic, domestic to romantic, romantic to moral, moral to social, social to professional, or professional to family. "
            "The new goal must create different scene types, different stakes, and a new emotional question. "
            "Once the new arc begins, the previous domain may remain as background pressure but should not stay the main plot unless a major reversal occurs. "
        )
    user = (
        "TASK=new_goal\n"
        f"Previous goal: {completed_goal}\n"
        f"Previous goal status: {goal_status}\n"
        f"Recent story paragraphs: {json.dumps(recent_story)}\n"
        f"World variables: {json.dumps(world_vars, sort_keys=True)}\n"
        f"Available characters: {json.dumps(character_context, sort_keys=True)}\n"
        f"Goal history: {json.dumps(goal_history)}\n"
        "Write a new story goal that follows from the recent story, raises or redirects the stakes, "
        "does not repeat past goals, and remains feasible to complete through future character actions. "
        "If the previous goal stalled before completion, choose a goal that expands the story and can be achieved from where the story is now instead of forcing the old goal to continue unchanged. "
        "Avoid repeating the previous goal verbatim or ending the story. "
        f"{domain_shift_lines}"
        f"{transition_lines}"
        f"Return JSON keys: {return_keys}."
    )
    return system, user
