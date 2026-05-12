from __future__ import annotations

from dataclasses import dataclass, replace

from pipeline import Config


@dataclass(frozen=True)
class AblationDefinition:
    name: str
    description: str
    max_plan_revisions: int | None = None
    enable_world_state_updates: bool | None = None
    enable_goal_shifts: bool | None = None
    stale_goal_steps: int | None = None
    arc_goal_steps: int | None = None

    def apply(self, cfg: Config) -> Config:
        updates: dict[str, object] = {}
        if self.max_plan_revisions is not None:
            updates["max_plan_revisions"] = self.max_plan_revisions
        if self.enable_world_state_updates is not None:
            updates["enable_world_state_updates"] = self.enable_world_state_updates
        if self.enable_goal_shifts is not None:
            updates["enable_goal_shifts"] = self.enable_goal_shifts
        if self.stale_goal_steps is not None:
            updates["stale_goal_steps"] = self.stale_goal_steps
        if self.arc_goal_steps is not None:
            updates["arc_goal_steps"] = self.arc_goal_steps
        return replace(cfg, **updates)


ABLATIONS: dict[str, AblationDefinition] = {
    "no_critic_revision": AblationDefinition(
        name="no_critic_revision",
        description="Disable critic-requested rewrites by forcing max plan revisions to zero.",
        max_plan_revisions=0,
    ),
    "no_world_state_updates": AblationDefinition(
        name="no_world_state_updates",
        description="Keep action selection and narration, but drop critic-proposed world state writes.",
        enable_world_state_updates=False,
    ),
    "no_goal_shifts": AblationDefinition(
        name="no_goal_shifts",
        description="Keep the initial story goal fixed and disable stale-goal refreshes and arc transitions.",
        enable_goal_shifts=False,
        stale_goal_steps=0,
        arc_goal_steps=0,
    ),
}


def get_ablation(name: str) -> AblationDefinition:
    key = name.strip().lower()
    if key not in ABLATIONS:
        available = ", ".join(sorted(ABLATIONS))
        raise ValueError(f"Unknown ablation '{name}'. Available ablations: {available}.")
    return ABLATIONS[key]


def list_ablation_names() -> list[str]:
    return sorted(ABLATIONS)
