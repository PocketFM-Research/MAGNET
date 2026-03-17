from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


@dataclass
class MemoryEntry:
    timestep: int
    character: str
    action: str
    narration: str
    reward: float
    world_before: dict[str, Any]
    world_after: dict[str, Any]


@dataclass
class StructuredMemory:
    entries: list[MemoryEntry] = field(default_factory=list)
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    _embed_model: Any = field(default=None, init=False)
    _index: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._embed_model = HuggingFaceEmbedding(model_name=self.embedding_model_name)

    def add(
        self,
        timestep: int,
        character: str,
        action: str,
        narration: str,
        reward: float,
        world_before: dict[str, Any],
        world_after: dict[str, Any],
    ) -> None:
        entry = MemoryEntry(
            timestep=timestep,
            character=character,
            action=action,
            narration=narration,
            reward=reward,
            world_before=dict(world_before),
            world_after=dict(world_after),
        )
        self.entries.append(entry)

        entry_idx = len(self.entries) - 1
        doc = Document(
            text=_entry_to_doc_text(entry),
            metadata={
                "entry_idx": entry_idx,
                "character": entry.character,
                "timestep": entry.timestep,
                "action": entry.action,
                "narration": entry.narration,
                "reward": entry.reward,
            },
        )

        if self._index is None:
            self._index = VectorStoreIndex.from_documents([doc], embed_model=self._embed_model)
        else:
            self._index.insert(doc)

    def retrieve(self, character: str, query: str, k: int = 3) -> list[str]:
        if not self.entries or self._index is None or k <= 0:
            return []

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="character",
                    operator=FilterOperator.EQ,
                    value=character,
                )
            ]
        )

        retriever = self._index.as_retriever(
            similarity_top_k=max(k * 3, k),
            filters=filters,
        )
        nodes = retriever.retrieve(query)
        if not nodes:
            return []

        max_step = max((e.timestep for e in self.entries), default=1)
        rescored: list[tuple[float, int]] = []
        seen_entry_ids: set[int] = set()

        for node in nodes:
            metadata = node.metadata or {}
            entry_idx = metadata.get("entry_idx")
            if not isinstance(entry_idx, int):
                continue
            if entry_idx in seen_entry_ids:
                continue
            if entry_idx < 0 or entry_idx >= len(self.entries):
                continue

            seen_entry_ids.add(entry_idx)
            entry = self.entries[entry_idx]
            base = float(node.score or 0.0)
            recency = entry.timestep / max_step
            reward_bonus = max(-1.0, min(1.0, entry.reward))
            score = base + (0.10 * recency) + (0.05 * reward_bonus)
            rescored.append((score, entry_idx))

        if not rescored:
            return []

        rescored.sort(key=lambda x: x[0], reverse=True)
        top_entries = [self.entries[idx] for _, idx in rescored[:k]]
        return [
            (
                f"t={entry.timestep} action={entry.action} reward={entry.reward:.2f} "
                f"narration={entry.narration}"
            )
            for entry in top_entries
        ]


def _entry_to_doc_text(entry: MemoryEntry) -> str:
    return (
        f"character={entry.character} action={entry.action} narration={entry.narration} "
        f"before={json.dumps(entry.world_before, sort_keys=True)} "
        f"after={json.dumps(entry.world_after, sort_keys=True)} "
        f"reward={entry.reward:.2f}"
    )
