from dataclasses import dataclass, field
from typing import Any


@dataclass
class CharacterProfile:
    name: str
    description: str


@dataclass
class CharacterDecision:
    action: str
    intent: str
    confidence: float
    revisions_used: int
    rationale: str


@dataclass
class StepResult:
    event_text: str
    reward: float
    done: bool
    info: dict[str, Any]


@dataclass
class SimulationState:
    world_vars: dict[str, Any] = field(default_factory=dict)
    timeline: list[str] = field(default_factory=list)
