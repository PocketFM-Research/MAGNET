from __future__ import annotations
import json
from typing import Any
import networkx as nx
from fables import FableDefinition, ActDefinition, PostRule, define_ant_and_dove_fable
from sim_types import StepResult


class WorldProxyEnv:
    def __init__(self, llm: object | None = None, fable: FableDefinition | None = None) -> None:
        self.world_graph = nx.DiGraph()
        self.llm = llm
        self.fable = fable or define_ant_and_dove_fable()

    def set_llm(self, llm: object | None) -> None:
        self.llm = llm

    def set_fable(self, fable: FableDefinition) -> None:
        self.fable = fable

    def reset(self, characters: list[str]) -> str:
        self.world_graph = nx.DiGraph()
        self.world_graph.add_node("world", kind="world")

        for character in characters:
            node_id = f"character:{character}"
            self.world_graph.add_node(node_id, kind="character", name=character)
            self.world_graph.add_edge("world", node_id, relation="contains")

        self._set_world_var("turn", 0)
        self._set_world_var("phase", "acts")
        self._set_world_var("current_act", 1)
        self._set_world_var("characters", list(characters))
        self._set_world_var("fable_name", self.fable.name)
        self._set_world_var("post_turns", 0)
        for key, value in self.fable.initial_world_vars.items():
            self._set_world_var(key, value)
        return self.fable.opening

    def get_world_vars(self) -> dict[str, Any]:
        world_vars: dict[str, Any] = {}
        for _, data in self.world_graph.nodes(data=True):
            if data.get("kind") == "state" and "key" in data:
                world_vars[str(data["key"])] = data.get("value")
        return world_vars

    def step(self, character: str, action: str) -> StepResult:
        self._set_world_var("turn", int(self._get_world_var("turn", 0)) + 1)
        actor = character.lower().strip()
        act = action.lower().strip()

        if self._get_world_var("phase") == "post":
            return self._step_post(actor, character, act, action)
        return self._step_acts(actor, character, act, action)

    def expected_actor(self) -> str | None:
        if self._get_world_var("phase") == "post":
            return None
        transition = self._get_current_act_definition()
        return transition.actor.lower().strip() if transition and transition.actor else None

    def _step_acts(self, actor: str, character: str, act: str, raw_action: str) -> StepResult:
        current_act = int(self._get_world_var("current_act", 1))
        transition = self._get_current_act_definition()
        if transition is None:
            return StepResult(event_text="No act transition.", reward=0.0, done=False, info={})

        can_advance = self._can_advance_transition(transition, actor, act)
        if can_advance:
            for key, value in transition.updates.items():
                self._set_world_var(key, value)

            if transition.act >= len(self.fable.acts):
                self._set_world_var("phase", "post")
                self._set_world_var("current_act", transition.act + 1)
            else:
                self._set_world_var("current_act", transition.act + 1)

            return StepResult(
                event_text=transition.event_text,
                reward=transition.reward,
                done=False,
                info={"act": transition.act},
            )

        return StepResult(
            event_text=f"Act {current_act} continues. {character} does '{raw_action}'.",
            reward=0.0,
            done=False,
            info={"act": current_act},
        )

    def _step_post(self, actor: str, character: str, act: str, raw_action: str) -> StepResult:
        self._set_world_var("post_turns", int(self._get_world_var("post_turns", 0)) + 1)
        matched_rule: PostRule | None = None
        for rule in self.fable.post_rules:
            actor_ok = rule.actor is None or actor == rule.actor.lower().strip()
            if not actor_ok:
                continue
            if not rule.requires_llm:
                matched_rule = rule
                break
            if self._llm_allows_progress(act, actor=actor, objective=rule.objective):
                matched_rule = rule
                break

        if matched_rule:
            event = matched_rule.event_text
            reward = matched_rule.reward
        else:
            event = f"Post-sim: {character} does '{raw_action}' as daily routine continues."
            reward = self.fable.post_fallback_reward

        done = int(self._get_world_var("post_turns", 0)) >= self.fable.post_done_after_turns
        return StepResult(event_text=event, reward=reward, done=done, info={"phase": "post"})

    def _can_advance_transition(self, transition: ActDefinition, actor: str, action: str) -> bool:
        actor_ok = transition.actor is None or actor == transition.actor.lower().strip()
        if not actor_ok:
            return False

        for key, expected in transition.preconditions.items():
            if self._get_world_var(key) != expected:
                return False

        if not transition.requires_llm:
            return True

        return self._llm_allows_progress(action, actor=actor, objective=transition.objective)

    def _get_current_act_definition(self) -> ActDefinition | None:
        current_act = int(self._get_world_var("current_act", 1))
        for transition in self.fable.acts:
            if transition.act == current_act:
                return transition
        return None

    def _llm_allows_progress(self, action: str, actor: str, objective: str | None) -> bool:
        if self.llm is None:
            return False

        if objective is None:
            return False

        system_prompt = (
            "You are a strict simulation transition judge. "
            "Given current world state and one action, decide if it semantically satisfies the stage objective. "
        )
        user_prompt = (
            f"TASK=stage_judge\n"
            f"Actor: {actor}\n"
            f"Objective: {objective}\n"
            f"World variables: {json.dumps(self.get_world_vars(), sort_keys=True)}\n"
            f"Action: {action}\n"
            "Return JSON keys: advance (boolean), confidence (0..1), reason (string)."
        )

        try:
            resp = self.llm.complete_json(system_prompt, user_prompt)
        except Exception:
            return False
        return bool(resp.get("advance", False))

    def _set_world_var(self, key: str, value: Any) -> None:
        node_id = f"state:{key}"
        self.world_graph.add_node(node_id, kind="state", key=key, value=value)
        self.world_graph.add_edge("world", node_id, relation="has_state")

    def _get_world_var(self, key: str, default: Any = None) -> Any:
        node_id = f"state:{key}"
        if not self.world_graph.has_node(node_id):
            return default
        return self.world_graph.nodes[node_id].get("value", default)
