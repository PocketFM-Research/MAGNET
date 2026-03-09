from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sim_types import CharacterProfile


@dataclass
class ActDefinition:
    act: int
    actor: str | None
    objective: str | None
    event_text: str
    reward: float
    preconditions: dict[str, Any] = field(default_factory=dict)
    updates: dict[str, Any] = field(default_factory=dict)
    requires_llm: bool = True


@dataclass
class PostRule:
    actor: str | None
    objective: str | None
    event_text: str
    reward: float
    requires_llm: bool = True


@dataclass
class FableDefinition:
    name: str
    goal: str
    opening: str
    characters: list[CharacterProfile]
    initial_world_vars: dict[str, Any]
    acts: list[ActDefinition]
    post_rules: list[PostRule] = field(default_factory=list)
    post_fallback_reward: float = 0.05
    post_done_after_turns: int = 2


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

    acts = [
        ActDefinition(
            act=1,
            actor="ant",
            objective=(
                "The ant's action should plausibly move the setup toward "
                "falling/slipping into river danger while foraging."
            ),
            event_text="Act 1: The ant slips into the river while searching for food.",
            reward=0.5,
            updates={
                "ant_in_water": True,
                "dove_safe": True,
            },
        ),
        ActDefinition(
            act=2,
            actor="dove",
            objective="The dove's action should plausibly rescue or save the ant from water danger right now.",
            event_text="Act 2: The dove rescues the ant with a leaf. A hunter appears and now threatens the dove.",
            reward=0.8,
            preconditions={"ant_in_water": True},
            updates={
                "ant_in_water": False,
                "ant_rescued": True,
                "hunter_present": True,
                "dove_endangered": True,
                "dove_safe": False,
            },
        ),
        ActDefinition(
            act=3,
            actor="ant",
            objective=None,
            event_text="Act 3: The hunter takes aim at the dove; danger escalates.",
            reward=0.4,
            preconditions={"hunter_present": True, "dove_endangered": True},
            updates={},
            requires_llm=False,
        ),
        ActDefinition(
            act=4,
            actor="ant",
            objective="The ant's action should plausibly prevent hunter harm and save the dove in this moment.",
            event_text="Act 4: The ant saves the dove and completes the fable arc.",
            reward=1.0,
            preconditions={"dove_endangered": True},
            updates={
                "hunter_present": False,
                "dove_endangered": False,
                "dove_safe": True,
                "ant_saved_dove": True,
            },
        ),
    ]

    post_rules = [
        PostRule(
            actor="ant",
            objective="The ant's action should reflect cautious, normal food-gathering routine after the main arc.",
            event_text="Post-sim: The ant gathers food carefully after surviving the river accident.",
            reward=0.2,
        ),
        PostRule(
            actor="dove",
            objective="The dove's action should reflect vigilant sky patrol routine after the hunter incident.",
            event_text="Post-sim: The dove patrols the sky cautiously after the hunter incident.",
            reward=0.2,
        ),
    ]

    return FableDefinition(
        name="ant_and_dove",
        goal="recreate_ant_and_dove_fable",
        opening="Act 1 begins: the ant searches for food near the river while the dove watches from a tree.",
        characters=characters,
        initial_world_vars={
            "ant_in_water": False,
            "ant_rescued": False,
            "hunter_present": False,
            "dove_endangered": False,
            "dove_safe": True,
            "ant_saved_dove": False,
        },
        acts=acts,
        post_rules=post_rules,
        post_fallback_reward=0.05,
        post_done_after_turns=2,
    )


def get_fable_definition(name: str) -> FableDefinition:
    key = name.lower().strip()
    if key in {"ant_and_dove", "the_ant_and_the_dove", "ant-dove"}:
        return define_ant_and_dove_fable()
    raise ValueError(f"Unknown fable definition '{name}'.")
