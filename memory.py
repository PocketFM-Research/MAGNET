from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryEntry:
    timestep: int
    character: str
    action: str
    event_text: str
    reward: float
    world_before: dict[str, Any]
    world_after: dict[str, Any]


@dataclass
class StructuredMemory:
    entries: list[MemoryEntry] = field(default_factory=list)

    def add(
        self,
        timestep: int,
        character: str,
        action: str,
        event_text: str,
        reward: float,
        world_before: dict[str, Any],
        world_after: dict[str, Any],
    ) -> None:
        self.entries.append(
            MemoryEntry(
                timestep=timestep,
                character=character,
                action=action,
                event_text=event_text,
                reward=reward,
                world_before=dict(world_before),
                world_after=dict(world_after),
            )
        )

    def retrieve(self, character: str, query: str, k: int = 3) -> list[str]:
        if not self.entries:
            return []

        query_tokens = _tokenize(query)
        scored: list[tuple[float, MemoryEntry]] = []

        for entry in self.entries:
            if entry.character != character:
                continue
            blob = (
                f"action={entry.action} event={entry.event_text} "
                f"before={json.dumps(entry.world_before, sort_keys=True)} "
                f"after={json.dumps(entry.world_after, sort_keys=True)}"
            )
            tokens = _tokenize(blob)
            overlap = len(query_tokens & tokens)
            norm = len(query_tokens | tokens) or 1
            score = overlap / norm
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]
        return [
            (
                f"t={entry.timestep} action={entry.action} reward={entry.reward:.2f} "
                f"event={entry.event_text}"
            )
            for _, entry in top
        ]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))
