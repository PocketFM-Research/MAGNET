from __future__ import annotations
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

    def reset(self, characters: list[str]) -> None:
        self.world_graph = nx.DiGraph()
        self.world_graph.add_node("world", kind="world")

        for character in characters:
            node_id = f"character:{character}"
            self.world_graph.add_node(node_id, kind="character", name=character)
            self.world_graph.add_edge("world", node_id, relation="contains")

        self._set_world_var("turn", 0)
        self._set_world_var("characters", list(characters))
        self._set_world_var("fable_name", self.fable.name)
        self._set_world_var("current_goal", self.fable.goal)
        self._set_world_var("goal_history", [self.fable.goal])
        for key, value in self.fable.initial_world_vars.items():
            self._set_world_var(key, value)
        return None

    def get_world_vars(self) -> dict[str, Any]:
        world_vars: dict[str, Any] = {}
        for _, data in self.world_graph.nodes(data=True):
            if data.get("kind") == "state" and "key" in data:
                world_vars[str(data["key"])] = data.get("value")
        return world_vars

    def step(
        self,
        character: str,
        action: str,
        advances_goal: bool = False,
        goal_reached: bool = False,
        world_updates: dict[str, Any] | None = None,
        progress_reason: str = "",
    ) -> StepResult:
        self._set_world_var("turn", int(self._get_world_var("turn", 0)) + 1)
        normalized_action = action.lower().strip()
        return self._step_story(
            character=character,
            action=normalized_action,
            raw_action=action,
            advances_goal=advances_goal,
            goal_reached=goal_reached,
            world_updates=world_updates,
            progress_reason=progress_reason,
        )

    def expected_actor(self) -> str | None:
        return None

    def _step_story(
        self,
        character: str,
        action: str,
        raw_action: str,
        advances_goal: bool = False,
        goal_reached: bool = False,
        world_updates: dict[str, Any] | None = None,
        progress_reason: str = "",
    ) -> StepResult:
        updates = world_updates if isinstance(world_updates, dict) else {}
        if not isinstance(updates, dict):
            updates = {}

        if advances_goal and not goal_reached and not updates:
            updates = {
                "progress_count": int(self._get_world_var("progress_count", 0)) + 1,
                "last_progress_actor": character,
                "last_progress_action": raw_action,
            }

        self._apply_world_updates(updates)

        if goal_reached:
            completed_goal = self.get_current_goal()
            self._set_world_var(self.fable.completion_key, True)
            return StepResult(
                event_text=f"Goal reached: {character} does '{raw_action}'.",
                reward=self.fable.progress_reward,
                done=False,
                info={
                    "goal_reached": True,
                    "advances_goal": True,
                    "goal_completed": True,
                    "completed_goal": completed_goal,
                    "progress_reason": progress_reason,
                },
            )

        if advances_goal:
            return StepResult(
                event_text=f"Story progresses toward the goal as {character} does '{raw_action}'.",
                reward=self.fable.progress_reward,
                done=False,
                info={
                    "goal_reached": False,
                    "advances_goal": True,
                    "progress_reason": progress_reason,
                },
            )

        return StepResult(
            event_text=f"Story continues. {character} does '{raw_action}'.",
            reward=self.fable.fallback_reward,
            done=False,
            info={
                "goal_reached": False,
                "advances_goal": False,
                "progress_reason": progress_reason,
            },
        )

    def _apply_world_updates(self, updates: dict[str, Any]) -> None:
        protected_keys = {"turn", "characters", "fable_name", "current_goal", "goal_history"}
        for key, value in updates.items():
            if not isinstance(key, str):
                continue
            if key in protected_keys:
                continue
            self._set_world_var(key, value)

    def get_current_goal(self) -> str:
        goal = self._get_world_var("current_goal", self.fable.goal)
        return str(goal) if goal else self.fable.goal

    def set_new_goal(self, new_goal: str) -> None:
        normalized_goal = new_goal.strip() or self.fable.goal
        previous_goal = self.get_current_goal()
        history = list(self._get_world_var("goal_history", [previous_goal]))
        history.append(normalized_goal)

        self._set_world_var("current_goal", normalized_goal)
        self._set_world_var("goal_history", history)
        self._set_world_var(self.fable.completion_key, False)

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
