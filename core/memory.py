"""
memory.py
---------
A lightweight vector-based shared memory for the agent team.

Why hand-rolled instead of a heavy vector DB?
- The design brief asks for "vector-based memory" and "context handoffs".
- A full Chroma/FAISS install is overkill for a 4-agent demo and hurts
  portability. So we implement a small bag-of-words TF embedding + cosine
  similarity. It is a real vector store (embed -> store -> similarity
  search), just dependency-free.

If you later want production-grade recall, swap SharedMemory's embed() and
recall() to call sentence-transformers + Chroma — the agent code won't
change because it only ever calls .add() and .recall().
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def embed(text: str) -> Counter:
    """Term-frequency 'embedding'. Sparse vector keyed by token."""
    return Counter(_tokenize(text))


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


@dataclass
class MemoryRecord:
    author: str          # which agent wrote it
    content: str         # the actual text
    vector: Counter = field(repr=False, default_factory=Counter)


class SharedMemory:
    """
    Append-only shared workspace. Every agent writes its output here, and
    any agent can recall the most relevant prior entries to inform its own
    step. This is how 'context handoff' physically happens in this system.
    """

    def __init__(self):
        self._records: list[MemoryRecord] = []

    def add(self, author: str, content: str) -> None:
        self._records.append(
            MemoryRecord(author=author, content=content, vector=embed(content))
        )

    def recall(
        self,
        query: str,
        k: int = 3,
        exclude_author: Optional[str] = None,
    ) -> list[MemoryRecord]:
        """Return the top-k records most similar to the query text."""
        qvec = embed(query)
        scored = [
            (cosine(qvec, r.vector), r)
            for r in self._records
            if exclude_author is None or r.author != exclude_author
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [r for score, r in scored[:k] if score > 0.0]

    def all_records(self) -> list[MemoryRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)
