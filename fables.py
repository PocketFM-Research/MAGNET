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


def define_corner_store_fable() -> FableDefinition:
    characters = [
        CharacterProfile(
            name="Maya",
            role="shop owner",
            traits=["guarded", "funny", "responsible"],
            goals=["protect her routines", "be honest about what she wants", "avoid another failed relationship"],
            fears=["misreading Omar", "upending her life for the wrong person"],
            abilities=["read people well", "keep conversations gentle", "notice small changes in others"],
            relationships={
                "Rafael": "father who wants her to stop hiding behind work",
                "Leah": "best friend who sees through her defenses",
                "Omar": "longtime neighbor she may already be in love with",
            },
            state={"at_store": True, "guarded": True, "hopeful": False},
            description=(
                "A 29-year-old woman who has spent so long keeping the corner store and her family together "
                "that she no longer trusts herself to want anything more personal."
            ),
        ),
        CharacterProfile(
            name="Rafael",
            role="father",
            traits=["warm", "perceptive", "old-fashioned"],
            goals=["see Maya build a life beyond work", "keep family ties honest", "nudge without controlling"],
            fears=["watching Maya settle for loneliness", "saying too much too soon"],
            abilities=["put people at ease", "notice emotional subtext", "offer advice without preaching"],
            relationships={
                "Maya": "daughter he loves and worries is holding back",
                "Leah": "trusted family friend",
                "Omar": "young man he quietly approves of",
            },
            state={"at_store": True, "watchful": True},
            description=(
                "A widower in his sixties who has learned that love often shows up as patience, timing, and "
                "the courage to say one truthful thing at the right moment."
            ),
        ),
        CharacterProfile(
            name="Leah",
            role="friend",
            traits=["witty", "direct", "loyal"],
            goals=["push Maya to be honest", "keep the friendship balanced", "prevent regret from masquerading as caution"],
            fears=["watching Maya miss her chance", "becoming the friend who meddles too hard"],
            abilities=["ask hard questions kindly", "lighten tense moments", "spot mutual attraction quickly"],
            relationships={
                "Maya": "best friend who needs occasional shoving",
                "Rafael": "father figure she respects",
                "Omar": "friend she trusts but teases relentlessly",
            },
            state={"available": True, "suspicious_of_chemistry": True},
            description=(
                "A public school counselor who has known both Maya and Omar for years and is tired of watching "
                "them confuse caution with maturity."
            ),
        ),
        CharacterProfile(
            name="Omar",
            role="neighbor",
            traits=["steady", "reserved", "deeply reliable"],
            goals=["tell Maya how he feels", "make the right choice about a job offer", "avoid pressuring her"],
            fears=["damaging their friendship", "leaving without saying what mattered"],
            abilities=["listen carefully", "show care through practical help", "stay calm under emotional pressure"],
            relationships={
                "Maya": "neighbor and closest friend he has quietly loved for years",
                "Rafael": "older mentor",
                "Leah": "friend who knows more than he says aloud",
            },
            state={"available": True, "job_offer_pending": True, "ready_to_leave": False},
            description=(
                "A building superintendent from two doors down who rarely speaks dramatically, but whose care "
                "for Maya has become impossible to hide now that he may leave the neighborhood."
            ),
        ),
    ]

    return FableDefinition(
        name="corner_store_last_week",
        goal="Maya must decide whether to risk a real romance with Omar before he gives an answer on a job offer that would take him away.",
        characters=characters,
        initial_world_vars={
            "goal_reached": False,
            "setting": "A neighborhood corner store and the apartments above it during one emotionally tense week.",
            "job_offer_deadline_days": 6, 
            "shared_history": "Maya and Omar have spent years circling around feelings neither of them has named plainly.",
            "current_tension": "Omar has a stable out-of-state job offer, and Maya has just realized losing him would feel personal, not merely practical.",
            "community_mood": "close-knit, observant, and aware that something unspoken is finally coming to a head",
        },
        progress_reward=0.5,
        fallback_reward=0.0,
        completion_key="goal_reached",
    )


def define_flood_rescue_fable() -> FableDefinition:
    characters = [
        CharacterProfile(
            name="Elena",
            role="fire captain",
            traits=["decisive", "calm", "protective"],
            goals=["get every passenger out of the stranded bus alive", "build a stable rescue line before the water rises higher", "make fast choices without wasting effort"],
            fears=["losing civilians to hesitation", "sending her crew into water she cannot control"],
            abilities=["lead rescues", "size up flood hazards", "coordinate under pressure"],
            relationships={
                "Nico": "trusted rescue swimmer",
                "Priya": "dispatcher she relies on for timing and updates",
                "Marcus": "civilian bus driver she needs to keep focused",
            },
            state={"on_scene": True, "injured": False, "commanding": True},
            description=(
                "A fire captain who has handled flood calls before and knows that the first clear plan often decides who lives."
            ),
        ),
        CharacterProfile(
            name="Nico",
            role="rescue swimmer",
            traits=["brave", "fast", "practical"],
            goals=["reach the stranded bus before the water rises into the cabin", "help passengers cross the rescue line fast", "support Elena's plan"],
            fears=["being pinned by debris", "losing footing in the current"],
            abilities=["swim strong currents", "handle ropes", "carry injured people through water"],
            relationships={
                "Elena": "captain he trusts completely",
                "Priya": "voice in his earpiece keeping him aligned",
                "Marcus": "civilian he may need to physically extract",
            },
            state={"suited_up": True, "ready": True},
            description=(
                "A rescue specialist who works best in cold water, bad light, and the kind of panic that makes everyone else slower."
            ),
        ),
        CharacterProfile(
            name="Priya",
            role="dispatcher",
            traits=["focused", "precise", "steady"],
            goals=["track flood changes", "route support efficiently", "keep the field team ahead of the next surge"],
            fears=["missing a critical update", "giving stale information"],
            abilities=["monitor emergency channels", "map routes quickly", "spot timing windows"],
            relationships={
                "Elena": "field commander she feeds information to",
                "Nico": "rescuer she keeps supplied with updates",
                "Marcus": "civilian caller she is trying to stabilize",
            },
            state={"at_command_post": True, "comms_clear": True},
            description=(
                "An emergency dispatcher who turns scattered reports, weather updates, and radio traffic into a rescue clock everyone else can act on."
            ),
        ),
        CharacterProfile(
            name="Marcus",
            role="bus driver",
            traits=["responsible", "shaken", "selfless"],
            goals=["keep the children on his bus safe", "empty the bus seat by seat without panic", "stay behind until the last passenger is out"],
            fears=["the water breaking through the door seals", "leaving a child behind"],
            abilities=["move people in orderly lines", "keep frightened passengers focused", "follow instructions when they are clear"],
            relationships={
                "Elena": "rescuer he is trying to trust",
                "Nico": "last line between the passengers and the river",
                "Priya": "voice on the radio guiding him through each minute",
            },
            state={"stranded": True, "injured": False, "responsible_for_children": True},
            description=(
                "A city bus driver stranded with children aboard after floodwater trapped his route at a low-water crossing, trying to sound steadier than he feels."
            ),
        ),
    ]

    return FableDefinition(
        name="flood_rescue_night",
        goal="Elena and her team must evacuate every passenger from a school bus stranded in floodwater at a washed-out crossing before Marcus sacrifices himself to save the last children aboard.",
        characters=characters,
        initial_world_vars={
            "goal_reached": False,
            "setting": "A school bus is stalled at a low-water crossing outside town after flash floodwater washes over the road.",
            "rescue_layout": "The bus is angled in fast-moving water near a washed-out shoulder. Rescuers must anchor a rope line from dry ground to the bus and move passengers out one at a time.",
            "road_state": "The asphalt beyond the bus has started to crumble where the shoulder gave way, so no one can simply drive or walk the bus out.",
            "bus_state": "A school bus with children on board is trapped in knee- to waist-high floodwater, with one door jammed and water beginning to seep into the first step.",
            "weather": "Heavy rain, poor visibility, and rising water.",
            "time_pressure": "Upstream runoff is still arriving, and another surge could lift or roll the bus within minutes.",
            "fatal_turn": "Marcus is the last adult inside the bus and may need to brace the rear emergency exit long enough for the last children to escape, knowing the current could take him when the bus shifts.",
        },
        progress_reward=0.5,
        fallback_reward=0.0,
        completion_key="goal_reached",
    )


def get_fable_definition(name: str) -> FableDefinition:
    key = name.lower().strip()
    if key in {"ant_and_dove", "the_ant_and_the_dove", "ant-dove"}:
        return define_ant_and_dove_fable()
    if key in {"corner_store_last_week", "corner_store", "maya_story", "maya store", "maya story"}:
        return define_corner_store_fable()
    if key in {"flood_rescue_night", "flood_rescue", "rescue_story", "action_story"}:
        return define_flood_rescue_fable()
    raise ValueError(f"Unknown fable definition '{name}'.")
