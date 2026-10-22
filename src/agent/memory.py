"""Multi-turn conversation memory with a character budget and JSON persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROLE_LABELS = {"user": "User", "assistant": "Assistant", "tool": "Tool"}


@dataclass(frozen=True)
class Turn:
    role: str
    text: str


class ConversationMemory:
    """Ordered turns that persist across the lifetime of an agent (and a file).

    ``render`` keeps the *most recent* turns that fit a character budget, so a
    small-context model always sees the latest exchange. If even the newest
    turn exceeds the budget, its tail is kept.
    """

    def __init__(self, turns: list[Turn] | None = None) -> None:
        self._turns: list[Turn] = list(turns or [])

    def __len__(self) -> int:
        return len(self._turns)

    @property
    def turns(self) -> tuple[Turn, ...]:
        return tuple(self._turns)

    def add(self, role: str, text: str) -> None:
        if role not in ROLE_LABELS:
            raise ValueError(f"role must be one of {tuple(ROLE_LABELS)}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("turn text must be a non-empty string")
        self._turns.append(Turn(role, text.strip()))

    def clear(self) -> None:
        self._turns.clear()

    def render(self, *, max_chars: int | None = None) -> str:
        lines = [f"{ROLE_LABELS[turn.role]}: {turn.text}" for turn in self._turns]
        if max_chars is None:
            return "\n".join(lines)
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        kept: list[str] = []
        used = 0
        for line in reversed(lines):
            cost = len(line) + (1 if kept else 0)
            if used + cost > max_chars:
                break
            kept.append(line)
            used += cost
        if not kept and lines:
            kept = [lines[-1][-max_chars:]]
        return "\n".join(reversed(kept))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps([asdict(turn) for turn in self._turns], indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ConversationMemory":
        memory = cls()
        for row in json.loads(Path(path).read_text(encoding="utf-8")):
            memory.add(row["role"], row["text"])
        return memory
