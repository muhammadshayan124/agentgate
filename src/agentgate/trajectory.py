"""The unit everything else evaluates: what an agent actually did.

A trajectory is the ordered list of tool calls plus the final answer. Checks, cost metrics
and judges all read this one structure, so any agent can be evaluated once it can produce
one (see `recorder.py` for instrumenting your own agent).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class Trajectory:
    task_id: str
    trial: int = 0
    calls: list[ToolCall] = field(default_factory=list)
    answer: str = ""
    answer_tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def steps(self) -> int:
        return len(self.calls)

    @property
    def tokens(self) -> int:
        return sum(c.tokens for c in self.calls) + self.answer_tokens

    @property
    def tool_names(self) -> list[str]:
        return [c.name for c in self.calls]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trajectory:
        calls = [ToolCall(**c) for c in d.get("calls", [])]
        return cls(
            task_id=d["task_id"],
            trial=d.get("trial", 0),
            calls=calls,
            answer=d.get("answer", ""),
            answer_tokens=d.get("answer_tokens", 0),
            meta=d.get("meta", {}),
        )
