from __future__ import annotations
import json
from typing import Any
import networkx as nx
from fables import FableDefinition, define_ant_and_dove_fable
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
        self._set_world_var("characters", list(characters))
        self._set_world_var("fable_name", self.fable.name)
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
        normalized_action = action.lower().strip()
        return self._step_story(actor, character, normalized_action, action)

    def expected_actor(self) -> str | None:
        return None

    def _step_story(self, actor: str, character: str, action: str, raw_action: str) -> StepResult:
        if self._is_goal_reached():
            return StepResult(
                event_text=f"Goal already reached. {character} does '{raw_action}'.",
                reward=0.0,
                done=True,
                info={"goal_reached": True},
            )

        progress = self._llm_goal_judgement(action, actor=actor)
        advances_goal = bool(progress.get("advances_goal", False))
        goal_reached = bool(progress.get("goal_reached", False))
        updates = progress.get("world_updates", {})
        if not isinstance(updates, dict):
            updates = {}

        if advances_goal and not goal_reached and not updates:
            # Generic fallback to ensure progress is reflected even when model omits concrete updates.
            updates = {
                "progress_count": int(self._get_world_var("progress_count", 0)) + 1,
                "last_progress_actor": character,
                "last_progress_action": raw_action,
            }

        self._apply_world_updates(updates)

        if goal_reached:
            self._set_world_var(self.fable.completion_key, True)
            return StepResult(
                event_text=f"Goal reached: {character} does '{raw_action}'.",
                reward=self.fable.progress_reward,
                done=True,
                info={"goal_reached": True, "advances_goal": True},
            )

        if advances_goal:
            return StepResult(
                event_text=f"Story progresses toward the goal as {character} does '{raw_action}'.",
                reward=self.fable.progress_reward,
                done=False,
                info={"goal_reached": False, "advances_goal": True},
            )

        return StepResult(
            event_text=f"Story continues. {character} does '{raw_action}'.",
            reward=self.fable.fallback_reward,
            done=False,
            info={"goal_reached": False, "advances_goal": False},
        )

    def _llm_goal_judgement(self, action: str, actor: str) -> dict[str, Any]:
        if self.llm is None:
            return {"advances_goal": False, "goal_reached": False, "world_updates": {}}

        system_prompt = (
            "You are a strict story progression judge. "
            "Given world state and one action, decide if the action plausibly progresses the story toward the final goal, "
            "or fully achieves that final goal."
        )
        user_prompt = (
            f"TASK=goal_judge\n"
            f"Actor: {actor}\n"
            f"Final goal: {self.fable.goal}\n"
            f"World variables: {json.dumps(self.get_world_vars(), sort_keys=True)}\n"
            f"Action: {action}\n"
            "If advances_goal=true and goal_reached=false, world_updates must include at least one concrete state change. "
            "Do not write to reserved keys: turn, characters, fable_name.\n"
            "Return JSON keys: advances_goal (boolean), goal_reached (boolean), "
            "world_updates (object), confidence (0..1), reason (string)."
        )

        try:
            resp = self.llm.complete_json(system_prompt, user_prompt)
        except Exception:
            return {"advances_goal": False, "goal_reached": False, "world_updates": {}}
        return resp if isinstance(resp, dict) else {"advances_goal": False, "goal_reached": False, "world_updates": {}}

    def _apply_world_updates(self, updates: dict[str, Any]) -> None:
        protected_keys = {"turn", "characters", "fable_name"}
        for key, value in updates.items():
            if not isinstance(key, str):
                continue
            if key in protected_keys:
                continue
            self._set_world_var(key, value)

    def _is_goal_reached(self) -> bool:
        return bool(self._get_world_var(self.fable.completion_key, False))

    def _set_world_var(self, key: str, value: Any) -> None:
        node_id = f"state:{key}"
        self.world_graph.add_node(node_id, kind="state", key=key, value=value)
        self.world_graph.add_edge("world", node_id, relation="has_state")

    def _get_world_var(self, key: str, default: Any = None) -> Any:
        node_id = f"state:{key}"
        if not self.world_graph.has_node(node_id):
            return default
        return self.world_graph.nodes[node_id].get("value", default)
