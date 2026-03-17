from dataclasses import dataclass, field
from typing import Any


@dataclass
class CharacterProfile:
    name: str
    role: str = ""
    traits: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    fears: list[str] = field(default_factory=list)
    abilities: list[str] = field(default_factory=list)
    relationships: dict[str, str] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def persona_text(self) -> str:
        parts: list[str] = []
        if self.description:
            parts.append(self.description.strip())
        if self.role:
            parts.append(f"Role: {self.role}.")
        if self.traits:
            parts.append(f"Traits: {', '.join(self.traits)}.")
        if self.goals:
            parts.append(f"Goals: {', '.join(self.goals)}.")
        if self.fears:
            parts.append(f"Fears: {', '.join(self.fears)}.")
        if self.abilities:
            parts.append(f"Abilities: {', '.join(self.abilities)}.")
        if self.relationships:
            rels = ", ".join(f"{name}={relation}" for name, relation in self.relationships.items())
            parts.append(f"Relationships: {rels}.")
        if self.state:
            flags = ", ".join(f"{k}={v}" for k, v in self.state.items())
            parts.append(f"Current state: {flags}.")
        return " ".join(parts) if parts else f"{self.name} has no profile details yet."

@dataclass
class CharacterDecision:
    character: str
    action: str
    intent: str
    confidence: float
    revisions_used: int
    rationale: str
    advances_goal: bool = False
    goal_reached: bool = False
    world_updates: dict[str, Any] = field(default_factory=dict)
    progress_reason: str = ""

@dataclass
class StepResult:
    event_text: str
    reward: float
    done: bool
    info: dict[str, Any]


@dataclass
class NarratedStep:
    paragraph: str
    included_indices: list[int] = field(default_factory=list)
    continuity_note: str = ""


@dataclass
class SimulationState:
    world_vars: dict[str, Any] = field(default_factory=dict)
    timeline: list[str] = field(default_factory=list)
