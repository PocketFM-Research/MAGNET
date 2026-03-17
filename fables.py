from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sim_types import CharacterProfile


@dataclass
class FableDefinition:
    name: str
    goal: str
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
        CharacterProfile(
            name="Spider",
            role="trickster",
            traits=["cunning", "patient", "observant"],
            goals=["gather information", "protect its web", "manipulate others subtly"],
            fears=["disturbance to its territory", "being attacked by predators"],
            abilities=["weave webs", "camouflage", "set traps"],
            relationships={"Ant": "neutral observer", "Dove": "neutral observer"},
            state={"web_intact": True, "alert": True},
            description=(
                "A quiet spider that studies every movement in the branches, guarding its web "
                "while using patience and subtle traps to influence events without revealing itself too soon."
            ),
        ),
        # CharacterProfile(
        #     name="Grasshopper",
        #     role="companion",
        #     traits=["optimistic", "spontaneous", "reckless"],
        #     goals=["seek adventure", "help friends", "enjoy life"],
        #     fears=["being trapped", "missing out on fun"],
        #     abilities=["jump long distances", "make distracting noise", "escape quickly"],
        #     relationships={"Ant": "friend", "Dove": "friend"},
        #     state={"energetic": True, "cautious": False},
        #     description=(
        #         "A lively grasshopper who leaps into trouble as quickly as into celebration, "
        #         "often helping friends with bold moves before thinking through the risk."
        #     ),
        # ),
        # CharacterProfile(
        #     name="Bee",
        #     role="protector",
        #     traits=["loyal", "alert", "hardworking"],
        #     goals=["protect hive", "gather resources", "maintain order"],
        #     fears=["threats to hive", "predators"],
        #     abilities=["sting", "fly quickly", "communicate via dance"],
        #     relationships={"Ant": "cautious ally", "Dove": "neutral"},
        #     state={"alert": True, "busy": True},
        #     description=(
        #         "A disciplined bee that balances speed with duty, always watching for danger "
        #         "while trying to keep the wider meadow safe and orderly."
        #     ),
        # ),
    ]

    return FableDefinition(
        name="ant_and_dove",
        goal="The ant saves the dove from the hunter.",
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
