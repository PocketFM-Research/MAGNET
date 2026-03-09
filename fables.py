from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sim_types import CharacterProfile


@dataclass
class FableDefinition:
    name: str
    goal: str
    opening: str
    characters: list[CharacterProfile]
    initial_world_vars: dict[str, Any]
    progress_reward: float = 0.5
    fallback_reward: float = 0.0
    completion_key: str = "goal_reached"


def define_ant_and_dove_fable() -> FableDefinition:
    characters = [
        CharacterProfile(
            name="Ant",
            role="helper",
            traits=["hardworking", "small", "persistent"],
            goals=["find food", "survive", "repay kindness"],
            fears=["drowning"],
            abilities=["bite", "crawl", "gather food"],
            relationships={"Dove": "grateful ally"},
            state={"injured": False, "wet": False},
            description="An observant ant who remembers favors and acts decisively under pressure.",
        ),
        CharacterProfile(
            name="Dove",
            role="protector",
            traits=["compassionate", "alert", "brave"],
            goals=["protect nearby creatures", "avoid the hunter"],
            fears=["hunter's arrows"],
            abilities=["fly", "spot danger", "carry twigs and leaves"],
            relationships={"Ant": "trusted friend"},
            state={"injured": False, "nest_safe": True},
            description="A watchful dove who intervenes quickly when others are in danger.",
        ),
    ]

    return FableDefinition(
        name="ant_and_dove",
        goal="The ant and dove survive danger through mutual help, and the ant ultimately saves the dove from the hunter.",
        opening="The ant searches for food near the river while the dove watches from a tree.",
        characters=characters,
        initial_world_vars={
            "goal_reached": False,
        },
        progress_reward=0.5,
        fallback_reward=0.0,
        completion_key="goal_reached",
    )


def get_fable_definition(name: str) -> FableDefinition:
    key = name.lower().strip()
    if key in {"ant_and_dove", "the_ant_and_the_dove", "ant-dove"}:
        return define_ant_and_dove_fable()
    raise ValueError(f"Unknown fable definition '{name}'.")
