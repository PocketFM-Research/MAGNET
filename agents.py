"""Minimal multi-step character agent with prompt chaining."""

from __future__ import annotations

from prompts import build_action_prompt, build_critic_prompt, build_intent_prompt
from sim_types import CharacterDecision, CharacterProfile


class CharacterAgent:
    def __init__(self, profile: CharacterProfile, llm: object) -> None:
        self.profile = profile
        self.llm = llm

    def decide_action(
        self,
        goal: str,
        world_vars: dict,
        memory_snippets: list[str],
        max_plan_revisions: int,
    ) -> CharacterDecision:
        intent_sys, intent_user = build_intent_prompt(
            self.profile.name,
            self.profile.description,
            goal,
            world_vars,
            memory_snippets,
        )
        intent_resp = self.llm.complete_json(intent_sys, intent_user)
        intent = str(intent_resp.get("intent", "advance the goal"))
        constraints_raw = intent_resp.get("constraints", [])
        constraints = [str(x) for x in constraints_raw] if isinstance(constraints_raw, list) else []

        revisions_used = 0
        feedback = None
        action = "look around"
        confidence = 0.4
        rationale = "fallback"

        while True:
            action_sys, action_user = build_action_prompt(
                self.profile.name,
                self.profile.description,
                intent,
                constraints,
                world_vars,
                memory_snippets,
                feedback,
            )
            action_resp = self.llm.complete_json(action_sys, action_user)
            action = str(action_resp.get("action", "look around")).strip() or "look around"
            confidence_value = action_resp.get("confidence", 0.4)
            confidence = float(confidence_value) if isinstance(confidence_value, (int, float, str)) else 0.4
            rationale = str(action_resp.get("rationale", ""))

            if world_vars.get("phase") == "post" or max_plan_revisions <= 0:
                revise = False
                feedback = ""
            else:
                critic_sys, critic_user = build_critic_prompt(self.profile.name, action, world_vars)
                critic_resp = self.llm.complete_json(critic_sys, critic_user)
                revise = bool(critic_resp.get("revise", False))
                feedback = str(critic_resp.get("feedback", ""))

            if not revise or revisions_used >= max_plan_revisions:
                break
            revisions_used += 1

        return CharacterDecision(
            action=action,
            intent=intent,
            confidence=confidence,
            revisions_used=revisions_used,
            rationale=rationale,
        )
