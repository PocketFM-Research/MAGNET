from __future__ import annotations
from dataclasses import dataclass
from agents import CharacterAgent, NarratorAgent
from environment import WorldProxyEnv
from llm import build_default_llm
from memory import StructuredMemory
from sim_types import CharacterProfile


@dataclass
class Config:
    goal: str = "create a compelling story that achieves the characters' goals"
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
        narrator = NarratorAgent(llm=self.llm)

        env.reset([c.name for c in characters])
        timeline: list[str] = []
        story: list[str] = []
        total_reward = 0.0

        for step in range(1, cfg.max_steps + 1):
            for agent in agents:
                world_before = env.get_world_vars()
                active_goal = env.get_current_goal()
                if not self._should_act_now(agent.profile.name, world_before, env):
                    continue
                query = f"goal={active_goal} world={world_before}"
                memory_snippets = self.memory.retrieve(agent.profile.name, query=query, k=cfg.rag_k)

                decision = None
                progress_feedback = None
                progress_revisions_used = 0

                progress = None
                while True:
                    decision = agent.decide_action(
                        goal=active_goal,
                        world_vars=world_before,
                        memory_snippets=memory_snippets,
                        max_plan_revisions=cfg.max_plan_revisions,
                        revision_feedback=progress_feedback,
                    )
                    progress = env.judge_action(
                        decision.action.lower().strip(),
                        actor=agent.profile.name.lower().strip(),
                    )
                    if bool(progress.get("advances_goal", False)) or progress_revisions_used >= cfg.max_plan_revisions:
                        break
                    progress_feedback = (
                        "The environment judge said this action does not advance the current goal. "
                        f"Reason: {str(progress.get('reason', '')).strip() or 'No reason provided.'} "
                        "Revise toward a more goal-relevant next action while staying plausible and in character."
                    )
                    progress_revisions_used += 1

                assert decision is not None
                result = env.step(agent.profile.name, decision.action, progress=progress)
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
                        f"t={step} goal={active_goal} actor={agent.profile.name} intent={decision.intent} "
                        f"action={decision.action} conf={decision.confidence:.2f} "
                        f"rev={decision.revisions_used + progress_revisions_used} -> {result.event_text}"
                    )
                )
                story.append(
                    narrator.narrate_step(
                        story_goal=active_goal,
                        recent_story=story[-3:],
                        actor=agent.profile.name,
                        intent=decision.intent,
                        action=decision.action,
                        event_text=result.event_text,
                        world_before=world_before,
                        world_after=world_after,
                    )
                )

                if result.info.get("goal_completed"):
                    completed_goal = str(result.info.get("completed_goal", active_goal))
                    new_goal = narrator.generate_next_goal(
                        completed_goal=completed_goal,
                        recent_story=story[-3:],
                        world_vars=world_after,
                    )
                    env.set_new_goal(new_goal)
                    rolled_world = env.get_world_vars()
                    timeline.append(
                        f"t={step} narrator=new_goal completed_goal={completed_goal} -> next_goal={new_goal}"
                    )
                    world_after = rolled_world

                if result.done:
                    return {
                        "done": True,
                        "steps": step,
                        "total_reward": total_reward,
                        "world_vars": world_after,
                        "timeline": timeline,
                        "story": story,
                    }

        return {
            "done": False,
            "steps": cfg.max_steps,
            "total_reward": total_reward,
            "world_vars": env.get_world_vars(),
            "timeline": timeline,
            "story": story,
        }

    @staticmethod
    def _should_act_now(character_name: str, world_vars: dict, env: WorldProxyEnv) -> bool:
        return True
