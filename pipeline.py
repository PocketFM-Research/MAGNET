from __future__ import annotations
from dataclasses import dataclass
from agents import CharacterAgent, NarratorAgent
from environment import WorldProxyEnv
from llm import build_default_llm
from sim_types import CharacterProfile


@dataclass
class Config:
    goal: str = "create a compelling story that achieves the characters' goals"
    max_steps: int = 8
    max_plan_revisions: int = 2
    use_rag: bool = False
    rag_k: int = 3


class Pipeline:
    def __init__(self, llm: object | None = None, use_rag: bool = False) -> None:
        self.llm = llm or build_default_llm()
        self.use_rag = use_rag
        self.memory = self._build_memory() if use_rag else None

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
            world_before = env.get_world_vars()
            active_goal = env.get_current_goal()
            proposed_actions = []

            for agent in agents:
                if not self._should_act_now(agent.profile.name, world_before, env):
                    continue
                if cfg.use_rag and self.memory is not None:
                    query = f"goal={active_goal} world={world_before}"
                    memory_snippets = self.memory.retrieve(agent.profile.name, query=query, k=cfg.rag_k)
                else:
                    memory_snippets = []

                decision = agent.decide_action(
                    goal=active_goal,
                    world_vars=world_before,
                    memory_snippets=memory_snippets,
                    recent_story=story[-4:],
                    max_plan_revisions=cfg.max_plan_revisions,
                )
                timeline.append(
                    (
                        f"t={step} goal={active_goal} proposed_actor={agent.profile.name} "
                        f"action={decision.action} conf={decision.confidence:.2f} "
                        f"rev={decision.revisions_used} advances_goal={decision.advances_goal} "
                        f"goal_reached={decision.goal_reached}"
                    )
                )
                proposed_actions.append(decision)

            if not proposed_actions:
                continue

            narrated_step = narrator.narrate_step(
                story_goal=active_goal,
                recent_story=story[-3:],
                world_before=world_before,
                proposals=proposed_actions,
            )

            included_indices: list[int] = []
            seen_indices: set[int] = set()
            for idx in narrated_step.included_indices:
                if idx in seen_indices:
                    continue
                seen_indices.add(idx)
                included_indices.append(idx)

            selected_actions = [proposed_actions[idx] for idx in included_indices]
            for idx, decision in enumerate(selected_actions):
                if decision.goal_reached:
                    selected_actions = selected_actions[: idx + 1]
                    break

            if not selected_actions:
                continue

            results = env.step_selected_actions(selected_actions)
            world_after = env.get_world_vars()
            step_reward = sum(result.reward for result in results)
            total_reward += step_reward

            selected_summary = ", ".join(
                f"{decision.character}: {decision.action}" for decision in selected_actions
            )
            timeline.append(f"t={step} narrator_selected={selected_summary}")
            timeline.append(f"t={step} story={narrated_step.paragraph}")
            story.append(narrated_step.paragraph)

            if cfg.use_rag and self.memory is not None:
                self.memory.add(
                    timestep=step,
                    characters=[decision.character for decision in selected_actions],
                    actions=[decision.action for decision in selected_actions],
                    narration=narrated_step.paragraph,
                    reward=step_reward,
                    world_before=world_before,
                    world_after=world_after,
                )

            for decision, result in zip(selected_actions, results):
                timeline.append(
                    (
                        f"t={step} canonical_actor={decision.character} action={decision.action} "
                        f"-> {result.event_text}"
                    )
                )

            completed_result = next((result for result in results if result.info.get("goal_completed")), None)
            if completed_result is not None:
                completed_goal = str(completed_result.info.get("completed_goal", active_goal))
                new_goal = narrator.generate_next_goal(
                    completed_goal=completed_goal,
                    recent_story=story[-3:],
                    world_vars=world_after,
                    character_context=self._build_character_context(characters),
                    goal_history=list(world_after.get("goal_history", [active_goal])),
                )
                env.set_new_goal(new_goal)
                world_after = env.get_world_vars()
                timeline.append(
                    f"t={step} narrator=new_goal completed_goal={completed_goal} -> next_goal={new_goal}"
                )

            if any(result.done for result in results):
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

    @staticmethod
    def _build_character_context(characters: list[CharacterProfile]) -> list[dict[str, object]]:
        context: list[dict[str, object]] = []
        for profile in characters:
            context.append(
                {
                    "name": profile.name,
                    "role": profile.role,
                    "traits": profile.traits,
                    "goals": profile.goals,
                    "abilities": profile.abilities,
                    "relationships": profile.relationships,
                    "state": profile.state,
                }
            )
        return context

    @staticmethod
    def _build_memory() -> object | None:
        from memory import StructuredMemory

        try:
            return StructuredMemory()
        except Exception:
            return None
