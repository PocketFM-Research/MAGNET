from __future__ import annotations
from dataclasses import dataclass
from agents import CharacterAgent, NarratorAgent
from environment import WorldProxyEnv
from llm import build_action_llm, build_critic_llm, build_default_llm, build_narrator_llm
from sim_types import CharacterProfile


@dataclass
class Config:
    goal: str = "create a compelling story that achieves the characters' goals"
    max_steps: int = 8
    max_plan_revisions: int = 2
    stale_goal_steps: int = 15
    arc_goal_steps: int = 40
    use_rag: bool = False
    rag_k: int = 3


class Pipeline:
    def __init__(
        self,
        llm: object | None = None,
        critic_llm: object | None = None,
        narrator_llm: object | None = None,
        action_llm: object | None = None,
        use_rag: bool = False,
    ) -> None:
        default_llm = llm or build_default_llm()
        self.llm = default_llm
        self.critic_llm = critic_llm or build_critic_llm(default_llm=default_llm)
        self.narrator_llm = narrator_llm or build_narrator_llm(default_llm=default_llm)
        self.action_llm = action_llm or build_action_llm(default_llm=self.critic_llm)
        self.use_rag = use_rag
        self.memory = self._build_memory() if use_rag else None

    def run(self, env: WorldProxyEnv, characters: list[CharacterProfile], cfg: Config | None = None) -> dict:
        cfg = cfg or Config()
        if hasattr(env, "set_llm"):
            env.set_llm(self.narrator_llm)
        agents = [
            CharacterAgent(profile=profile, action_llm=self.action_llm, critic_llm=self.critic_llm)
            for profile in characters
        ]
        narrator = NarratorAgent(llm=self.narrator_llm)

        env.reset([c.name for c in characters])
        timeline: list[str] = []
        story: list[str] = []
        total_reward = 0.0
        goal_assigned_step = 1

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
                goal_assigned_step = self._refresh_stale_goal_if_needed(
                    env=env,
                    narrator=narrator,
                    characters=characters,
                    story=story,
                    timeline=timeline,
                    step=step,
                    goal_assigned_step=goal_assigned_step,
                    stale_goal_steps=cfg.stale_goal_steps,
                    active_goal=active_goal,
                    world_after=world_before,
                )
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
                goal_assigned_step = self._refresh_stale_goal_if_needed(
                    env=env,
                    narrator=narrator,
                    characters=characters,
                    story=story,
                    timeline=timeline,
                    step=step,
                    goal_assigned_step=goal_assigned_step,
                    stale_goal_steps=cfg.stale_goal_steps,
                    active_goal=active_goal,
                    world_after=world_before,
                )
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
            arc_shift_due = self._is_arc_shift_due(step=step, arc_goal_steps=cfg.arc_goal_steps)
            if arc_shift_due:
                previous_goal = (
                    str(completed_result.info.get("completed_goal", active_goal))
                    if completed_result is not None
                    else active_goal
                )
                goal_status = (
                    "completed"
                    if completed_result is not None
                    else (
                        f"overall story reached step {step}; close this arc and generate "
                        "a successor conflict in a different life domain"
                    )
                )
                arc_shift_step = self._force_arc_goal_shift(
                    env=env,
                    narrator=narrator,
                    characters=characters,
                    story=story,
                    timeline=timeline,
                    step=step,
                    previous_goal=previous_goal,
                    goal_status=goal_status,
                    world_after=world_after,
                )
                if arc_shift_step is not None:
                    goal_assigned_step = arc_shift_step
                    completed_result = None
            if completed_result is not None:
                completed_goal = str(completed_result.info.get("completed_goal", active_goal))
                goal_transition = narrator.generate_next_goal(
                    completed_goal=completed_goal,
                    recent_story=story[-3:],
                    world_vars=world_after,
                    character_context=self._build_character_context(characters),
                    goal_history=list(world_after.get("goal_history", [active_goal])),
                )
                new_goal = goal_transition.get("goal", "").strip()
                if not new_goal or new_goal == completed_goal:
                    timeline.append(
                        f"t={step} narrator=new_goal_failed completed_goal={completed_goal}"
                    )
                    continue
                self._apply_goal_transition(
                    env=env,
                    world_after=world_after,
                    previous_goal=completed_goal,
                    goal_transition=goal_transition,
                )
                self._append_goal_transition_story(
                    story=story,
                    timeline=timeline,
                    step=step,
                    goal_transition=goal_transition,
                )
                env.set_new_goal(new_goal)
                world_after = env.get_world_vars()
                goal_assigned_step = step + 1
                timeline.append(
                    self._format_goal_transition_timeline(
                        step=step,
                        event_name="new_goal",
                        previous_goal=completed_goal,
                        goal_transition=goal_transition,
                    )
                )
            elif not arc_shift_due:
                goal_assigned_step = self._refresh_stale_goal_if_needed(
                    env=env,
                    narrator=narrator,
                    characters=characters,
                    story=story,
                    timeline=timeline,
                    step=step,
                    goal_assigned_step=goal_assigned_step,
                    stale_goal_steps=cfg.stale_goal_steps,
                    active_goal=active_goal,
                    world_after=world_after,
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

    def _refresh_stale_goal_if_needed(
        self,
        env: WorldProxyEnv,
        narrator: NarratorAgent,
        characters: list[CharacterProfile],
        story: list[str],
        timeline: list[str],
        step: int,
        goal_assigned_step: int,
        stale_goal_steps: int,
        active_goal: str,
        world_after: dict,
    ) -> int:
        if stale_goal_steps <= 0:
            return goal_assigned_step

        steps_since_assignment = step - goal_assigned_step + 1
        if steps_since_assignment < stale_goal_steps:
            return goal_assigned_step

        goal_history = list(world_after.get("goal_history", [active_goal]))
        try:
            goal_transition = narrator.generate_next_goal(
                completed_goal=active_goal,
                recent_story=story[-3:],
                world_vars=world_after,
                character_context=self._build_character_context(characters),
                goal_history=goal_history,
                goal_status=(
                    f"not achieved after {steps_since_assignment} steps since assignment; "
                    "replace it with a feasible goal for the current story state"
                ),
            )
        except Exception:
            return goal_assigned_step

        new_goal = goal_transition.get("goal", "").strip()
        if not new_goal or new_goal == active_goal:
            return goal_assigned_step

        self._apply_goal_transition(
            env=env,
            world_after=world_after,
            previous_goal=active_goal,
            goal_transition=goal_transition,
        )
        self._append_goal_transition_story(
            story=story,
            timeline=timeline,
            step=step,
            goal_transition=goal_transition,
        )
        env.set_new_goal(new_goal)
        timeline.append(
            self._format_goal_transition_timeline(
                step=step,
                event_name="stale_goal_refresh",
                previous_goal=active_goal,
                goal_transition=goal_transition,
                steps_since_assignment=steps_since_assignment,
            )
        )
        return step + 1

    def _force_arc_goal_shift(
        self,
        env: WorldProxyEnv,
        narrator: NarratorAgent,
        characters: list[CharacterProfile],
        story: list[str],
        timeline: list[str],
        step: int,
        previous_goal: str,
        goal_status: str,
        world_after: dict,
    ) -> int | None:
        goal_history = list(world_after.get("goal_history", [previous_goal]))
        previous_goal_domain = str(world_after.get("current_goal_domain", "")).strip()
        try:
            goal_transition = narrator.generate_next_goal(
                completed_goal=previous_goal,
                recent_story=story[-3:],
                world_vars=world_after,
                character_context=self._build_character_context(characters),
                goal_history=goal_history,
                goal_status=goal_status,
                force_domain_shift=True,
                previous_goal_domain=previous_goal_domain,
            )
        except Exception:
            return None

        new_goal = goal_transition.get("goal", "").strip()
        if not new_goal or new_goal == previous_goal:
            timeline.append(f"t={step} narrator=arc_goal_refresh_failed previous_goal={previous_goal}")
            return None

        self._apply_goal_transition(
            env=env,
            world_after=world_after,
            previous_goal=previous_goal,
            goal_transition=goal_transition,
        )
        self._append_goal_transition_story(
            story=story,
            timeline=timeline,
            step=step,
            goal_transition=goal_transition,
        )
        env.set_new_goal(new_goal)
        timeline.append(
            self._format_goal_transition_timeline(
                step=step,
                event_name="arc_goal_refresh",
                previous_goal=previous_goal,
                goal_transition=goal_transition,
            )
        )
        return step + 1

    @staticmethod
    def _is_arc_shift_due(step: int, arc_goal_steps: int) -> bool:
        return arc_goal_steps > 0 and step % arc_goal_steps == 0

    @staticmethod
    def _apply_goal_transition(
        env: WorldProxyEnv,
        world_after: dict,
        previous_goal: str,
        goal_transition: dict[str, str],
    ) -> None:
        next_goal = goal_transition.get("goal", "").strip()
        if not next_goal:
            return

        previous_goal_domain = str(world_after.get("current_goal_domain", "")).strip()
        goal_domain = goal_transition.get("goal_domain", "").strip()
        rationale = goal_transition.get("rationale", "").strip()
        updates = {
            "previous_goal": previous_goal,
            "previous_goal_domain": previous_goal_domain,
            "current_goal_domain": goal_domain,
            "goal_shift_rationale": rationale,
        }
        env.apply_world_updates(updates)

    @staticmethod
    def _append_goal_transition_story(
        story: list[str],
        timeline: list[str],
        step: int,
        goal_transition: dict[str, str],
    ) -> None:
        story_paragraph = goal_transition.get("transition_paragraph", "").strip()
        if not story_paragraph:
            return

        story.append(story_paragraph)
        timeline.append(f"t={step} transition_story={story_paragraph}")

    @staticmethod
    def _format_goal_transition_timeline(
        step: int,
        event_name: str,
        previous_goal: str,
        goal_transition: dict[str, str],
        steps_since_assignment: int | None = None,
    ) -> str:
        details = [
            f"t={step}",
            f"narrator={event_name}",
        ]
        if steps_since_assignment is not None:
            details.append(f"steps_since_assignment={steps_since_assignment}")
        details.extend(
            [
                f"previous_goal={previous_goal}",
                f"transition_paragraph={goal_transition.get('transition_paragraph', '').strip()}",
                f"goal_domain={goal_transition.get('goal_domain', '').strip()}",
                f"next_goal={goal_transition.get('goal', '').strip()}",
            ]
        )
        return " ".join(details)

    @staticmethod
    def _build_memory() -> object | None:
        from memory import StructuredMemory

        try:
            return StructuredMemory()
        except Exception:
            return None
