from __future__ import annotations
import json
from typing import Any
import networkx as nx
from sim_types import StepResult

class WorldProxyEnv:
    def __init__(self, llm: object | None = None) -> None:
        self.world_graph = nx.DiGraph()
        self.llm = llm

    def set_llm(self, llm: object | None) -> None:
        self.llm = llm

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
        self._set_world_var("ant_in_water", False)
        self._set_world_var("ant_rescued", False)
        self._set_world_var("hunter_present", False)
        self._set_world_var("dove_endangered", False)
        self._set_world_var("dove_safe", True)
        self._set_world_var("ant_saved_dove", False)
        self._set_world_var("post_turns", 0)
        return "Act 1 begins: the ant searches for food near the river while the dove watches from a tree."

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

    def _step_acts(self, actor: str, character: str, act: str, raw_action: str) -> StepResult:
        current_act = int(self._get_world_var("current_act", 1))

        if current_act == 1:
            if actor == "ant" and self._llm_allows_progress(act, actor=actor, stage="act1"):
                self._set_world_var("ant_in_water", True)
                self._set_world_var("dove_safe", True)
                self._set_world_var("current_act", 2)
                return StepResult(
                    event_text="Act 1: The ant slips into the river while searching for food.",
                    reward=0.5,
                    done=False,
                    info={"act": 1},
                )
            return StepResult(
                event_text=f"Act 1 continues. {character} does '{raw_action}'.",
                reward=0.0,
                done=False,
                info={"act": 1},
            )

        if current_act == 2:
            if (
                actor == "dove"
                and bool(self._get_world_var("ant_in_water", False))
                and self._llm_allows_progress(act, actor=actor, stage="act2")
            ):
                self._set_world_var("ant_in_water", False)
                self._set_world_var("ant_rescued", True)
                self._set_world_var("hunter_present", True)
                self._set_world_var("dove_endangered", True)
                self._set_world_var("dove_safe", False)
                self._set_world_var("current_act", 3)
                return StepResult(
                    event_text="Act 2: The dove rescues the ant with a leaf. A hunter appears and now threatens the dove.",
                    reward=0.8,
                    done=False,
                    info={"act": 2},
                )
            return StepResult(
                event_text=f"Act 2 continues. {character} does '{raw_action}'.",
                reward=0.0,
                done=False,
                info={"act": 2},
            )

        if current_act == 3:
            if bool(self._get_world_var("hunter_present", False)) and bool(self._get_world_var("dove_endangered", False)):
                self._set_world_var("current_act", 4)
                return StepResult(
                    event_text="Act 3: The hunter takes aim at the dove; danger escalates.",
                    reward=0.4,
                    done=False,
                    info={"act": 3},
                )
            return StepResult(
                event_text=f"Act 3 continues. {character} does '{raw_action}'.",
                reward=0.0,
                done=False,
                info={"act": 3},
            )

        if current_act == 4:
            if (
                actor == "ant"
                and bool(self._get_world_var("dove_endangered", False))
                and self._llm_allows_progress(act, actor=actor, stage="act4")
            ):
                self._set_world_var("hunter_present", False)
                self._set_world_var("dove_endangered", False)
                self._set_world_var("dove_safe", True)
                self._set_world_var("ant_saved_dove", True)
                self._set_world_var("phase", "post")
                self._set_world_var("current_act", 5)
                return StepResult(
                    event_text="Act 4: The ant saves the dove and completes the fable arc.",
                    reward=1.0,
                    done=False,
                    info={"act": 4},
                )
            return StepResult(
                event_text=f"Act 4 continues. {character} does '{raw_action}'.",
                reward=0.0,
                done=False,
                info={"act": 4},
            )

        return StepResult(event_text="No act transition.", reward=0.0, done=False, info={})

    def _step_post(self, actor: str, character: str, act: str, raw_action: str) -> StepResult:
        self._set_world_var("post_turns", int(self._get_world_var("post_turns", 0)) + 1)

        if actor == "ant" and self._llm_allows_progress(act, actor=actor, stage="post_ant"):
            event = "Post-sim: The ant gathers food carefully after surviving the river accident."
            reward = 0.2
        elif actor == "dove" and self._llm_allows_progress(act, actor=actor, stage="post_dove"):
            event = "Post-sim: The dove patrols the sky cautiously after the hunter incident."
            reward = 0.2
        else:
            event = f"Post-sim: {character} does '{raw_action}' as daily routine continues."
            reward = 0.05

        done = int(self._get_world_var("post_turns", 0)) >= 2
        return StepResult(event_text=event, reward=reward, done=done, info={"phase": "post"})

    def _llm_allows_progress(self, action: str, actor: str, stage: str) -> bool:
        if self.llm is None:
            return False

        objectives = {
            "act1": "The ant's action should plausibly move the setup toward falling/slipping into river danger while foraging.",
            "act2": "The dove's action should plausibly rescue or save the ant from water danger right now.",
            "act4": "The ant's action should plausibly prevent hunter harm and save the dove in this moment.",
            "post_ant": "The ant's action should reflect cautious, normal food-gathering routine after the main arc.",
            "post_dove": "The dove's action should reflect vigilant sky patrol routine after the hunter incident.",
        }
        objective = objectives.get(stage)
        if objective is None:
            return False

        system_prompt = (
            "You are a strict simulation transition judge. "
            "Given current world state and one action, decide if it semantically satisfies the stage objective. "
        )
        user_prompt = (
            f"TASK=stage_judge\n"
            f"Stage: {stage}\n"
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
