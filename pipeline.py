from __future__ import annotations

from dataclasses import dataclass

from agents import CharacterAgent
from environment import WorldProxyEnv
from llm import build_default_llm
from memory import StructuredMemory
from sim_types import CharacterProfile


@dataclass
class Config:
    goal: str = "recreate_ant_and_dove_fable"
    max_steps: int = 8
    max_plan_revisions: int = 2
    rag_k: int = 3


class Pipeline:
    def __init__(self, llm: object | None = None) -> None:
        self.llm = llm or build_default_llm()
        self.memory = StructuredMemory()

    def run(self, env: WorldProxyEnv, characters: list[CharacterProfile], cfg: Config | None = None) -> dict:
        cfg = cfg or Config()
        if hasattr(env, "set_llm"):
            env.set_llm(self.llm)
        agents = [CharacterAgent(profile=profile, llm=self.llm) for profile in characters]

        opening = env.reset([c.name for c in characters])
        timeline: list[str] = [opening]
        total_reward = 0.0

        for step in range(1, cfg.max_steps + 1):
            for agent in agents:
                world_before = env.get_world_vars()
                if not self._should_act_now(agent.profile.name, world_before):
                    continue
                query = f"goal={cfg.goal} world={world_before}"
                memory_snippets = self.memory.retrieve(agent.profile.name, query=query, k=cfg.rag_k)

                decision = agent.decide_action(
                    goal=cfg.goal,
                    world_vars=world_before,
                    memory_snippets=memory_snippets,
                    max_plan_revisions=cfg.max_plan_revisions,
                )
                result = env.step(agent.profile.name, decision.action)
                world_after = env.get_world_vars()

                self.memory.add(
                    timestep=step,
                    character=agent.profile.name,
                    action=decision.action,
                    event_text=result.event_text,
                    reward=result.reward,
                    world_before=world_before,
                    world_after=world_after,
                )

                total_reward += result.reward
                timeline.append(
                    (
                        f"t={step} actor={agent.profile.name} intent={decision.intent} "
                        f"action={decision.action} conf={decision.confidence:.2f} "
                        f"rev={decision.revisions_used} -> {result.event_text}"
                    )
                )

                if result.done:
                    return {
                        "done": True,
                        "steps": step,
                        "total_reward": total_reward,
                        "world_vars": world_after,
                        "timeline": timeline,
                    }

        return {
            "done": False,
            "steps": cfg.max_steps,
            "total_reward": total_reward,
            "world_vars": env.get_world_vars(),
            "timeline": timeline,
        }

    @staticmethod
    def _should_act_now(character_name: str, world_vars: dict) -> bool:
        """Reduce calls by selecting one main actor per act; keep both in post-sim."""
        if world_vars.get("phase") == "post":
            return True

        act = int(world_vars.get("current_act", 1))
        actor_by_act = {
            1: "ant",
            2: "dove",
            3: "ant",
            4: "ant",
        }
        expected = actor_by_act.get(act)
        if expected is None:
            return True
        return character_name.lower() == expected
