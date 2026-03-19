from __future__ import annotations
from prompts import (
    build_action_prompt,
    build_critic_prompt,
    build_new_goal_prompt,
    build_narrator_prompt,
)
from sim_types import CharacterDecision, CharacterProfile, NarratedStep


class CharacterAgent:
    def __init__(self, profile: CharacterProfile, llm: object) -> None:
        self.profile = profile
        self.llm = llm

    def decide_action(
        self,
        goal: str,
        world_vars: dict,
        memory_snippets: list[str],
        recent_story: list[str],
        max_plan_revisions: int,
        revision_feedback: str | None = None,
    ) -> CharacterDecision:
        persona = self.profile.persona_text()
        world_knowledge = self._build_world_knowledge(world_vars)
        last_scene_summary, previous_scene_summary = self._build_recent_history(recent_story)
        revisions_used = 0
        feedback = revision_feedback
        action = "look around"
        confidence = 0.4
        rationale = "fallback"
        advances_goal = False
        goal_reached = False
        world_updates: dict = {}
        progress_reason = ""

        while True:
            action_sys, action_user = build_action_prompt(
                self.profile.name,
                persona,
                goal,
                world_vars,
                memory_snippets,
                world_knowledge,
                last_scene_summary,
                previous_scene_summary,
                feedback,
            )
            action_resp = self.llm.complete_json(action_sys, action_user)
            action = str(action_resp.get("action", "look around")).strip() or "look around"
            confidence_value = action_resp.get("confidence", 0.4)
            confidence = float(confidence_value) if isinstance(confidence_value, (int, float, str)) else 0.4
            rationale = str(action_resp.get("rationale", ""))

            critic_sys, critic_user = build_critic_prompt(
                self.profile.name,
                persona,
                action,
                goal,
                world_vars,
            )
            critic_resp = self.llm.complete_json(critic_sys, critic_user)
            revise = bool(critic_resp.get("revise", False)) if max_plan_revisions > 0 else False
            feedback = str(critic_resp.get("feedback", ""))
            advances_goal = bool(critic_resp.get("advances_goal", False))
            goal_reached = bool(critic_resp.get("goal_reached", False))
            updates_raw = critic_resp.get("world_updates", {})
            world_updates = updates_raw if isinstance(updates_raw, dict) else {}
            progress_reason = str(critic_resp.get("reason", ""))

            if not revise or revisions_used >= max_plan_revisions:
                break
            revisions_used += 1

        return CharacterDecision(
            character=self.profile.name,
            action=action,
            confidence=confidence,
            revisions_used=revisions_used,
            rationale=rationale,
            advances_goal=advances_goal,
            goal_reached=goal_reached,
            world_updates=world_updates,
            progress_reason=progress_reason,
        )

    def _build_world_knowledge(self, world_vars: dict) -> list[str]:
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

        lowered_name = self.profile.name.lower()
        for key in sorted(world_vars):
            if key in seen_keys or len(facts) >= 5:
                continue
            if key in ignored_keys:
                continue
            value = world_vars[key]
            value_text = str(value).lower()
            key_text = key.lower()
            if lowered_name in key_text or lowered_name in value_text:
                facts.append(f"{key}={world_vars[key]}")
                seen_keys.add(key)

        for key in sorted(world_vars):
            if key in seen_keys or len(facts) >= 5:
                continue
            if key in ignored_keys:
                continue
            value = world_vars[key]
            key_text = key.lower()
            value_text = str(value).lower()
            if any(token in key_text for token in ("action", "event", "actor", "status", "state")):
                facts.append(f"{key}={world_vars[key]}")
                seen_keys.add(key)
                continue
            if any(name.lower() in value_text for name in self.profile.relationships) or self.profile.name.lower() in value_text:
                facts.append(f"{key}={world_vars[key]}")
                seen_keys.add(key)

        for key in sorted(world_vars):
            if key in seen_keys or len(facts) >= 5:
                continue
            if key in ignored_keys:
                continue
            value = world_vars[key]
            if isinstance(value, str) and value.strip():
                facts.append(f"{key}={value}")
                seen_keys.add(key)

        if not facts:
            facts.append("No relevant character events are currently known.")

        return facts[:5]

    @staticmethod
    def _build_recent_history(recent_story: list[str]) -> tuple[str, str]:
        if not recent_story:
            return "No prior scene has been narrated yet.", "No earlier scene is available."

        last_scene = recent_story[-1].strip() or "No details available."
        previous_scene = recent_story[-2].strip() if len(recent_story) > 1 else "No earlier scene is available."
        if not previous_scene:
            previous_scene = "No details available."
        return last_scene, previous_scene


class NarratorAgent:
    def __init__(self, llm: object) -> None:
        self.llm = llm

    def narrate_step(
        self,
        story_goal: str,
        recent_story: list[str],
        world_before: dict,
        proposals: list[CharacterDecision],
    ) -> NarratedStep:
        proposal_payload = [
            {
                "index": idx,
                "character": proposal.character,
                "action": proposal.action,
                "confidence": proposal.confidence,
                "rationale": proposal.rationale,
                "advances_goal": proposal.advances_goal,
                "goal_reached": proposal.goal_reached,
                "world_updates": proposal.world_updates,
                "progress_reason": proposal.progress_reason,
            }
            for idx, proposal in enumerate(proposals)
        ]
        narrator_sys, narrator_user = build_narrator_prompt(
            story_goal=story_goal,
            recent_story=recent_story,
            world_before=world_before,
            proposals=proposal_payload,
        )
        try:
            narrator_resp = self.llm.complete_json(narrator_sys, narrator_user)
        except Exception:
            paragraph = "; ".join(
                f"{proposal.character} {proposal.action}" for proposal in proposals[:2]
            ).strip()
            return NarratedStep(
                paragraph=paragraph or "Nothing noteworthy happened.",
                included_indices=list(range(min(len(proposals), 2))),
            )

        included_raw = narrator_resp.get("included_indices", [])
        included_indices = [
            idx for idx in included_raw if isinstance(idx, int) and 0 <= idx < len(proposals)
        ] if isinstance(included_raw, list) else []
        paragraph = str(narrator_resp.get("paragraph", "")).strip()
        continuity_note = str(narrator_resp.get("continuity_note", "")).strip()
        if not included_indices and proposals:
            included_indices = [0]
        if not paragraph and included_indices:
            paragraph = "; ".join(
                f"{proposals[idx].character} {proposals[idx].action}" for idx in included_indices
            )
        return NarratedStep(
            paragraph=paragraph or "Nothing noteworthy happened.",
            included_indices=included_indices,
            continuity_note=continuity_note,
        )

    def generate_next_goal(
        self,
        completed_goal: str,
        recent_story: list[str],
        world_vars: dict,
        character_context: list[dict],
        goal_history: list[str],
    ) -> str:
        goal_sys, goal_user = build_new_goal_prompt(
            completed_goal=completed_goal,
            recent_story=recent_story,
            world_vars=world_vars,
            character_context=character_context,
            goal_history=goal_history,
        )
        goal_resp = self.llm.complete_json(goal_sys, goal_user)

        next_goal = str(goal_resp.get("goal", "")).strip()
        return next_goal 
