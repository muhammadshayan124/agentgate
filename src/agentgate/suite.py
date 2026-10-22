"""Task suites: what the agent is asked to do and how each attempt is judged."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agentgate.checks import Check, build_check


@dataclass
class Task:
    id: str
    prompt: str
    params: dict[str, Any] = field(default_factory=dict)
    check_specs: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @property
    def checks(self) -> list[Check]:
        return [build_check(spec, self.params) for spec in self.check_specs]


@dataclass
class Suite:
    name: str
    tasks: list[Task]

    def __post_init__(self) -> None:
        ids = [t.id for t in self.tasks]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate task ids: {dupes}")
        for t in self.tasks:  # fail at load time, not mid-run
            t.checks  # noqa: B018

    @classmethod
    def load(cls, path: str | Path) -> Suite:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        tasks = [
            Task(
                id=t["id"],
                prompt=t.get("prompt", ""),
                params=t.get("params", {}),
                check_specs=t.get("checks", []),
                tags=t.get("tags", []),
            )
            for t in raw["tasks"]
        ]
        return cls(name=raw.get("name", Path(path).stem), tasks=tasks)

    def to_yaml(self) -> str:
        doc = {
            "name": self.name,
            "tasks": [
                {"id": t.id, "prompt": t.prompt, "params": t.params, "tags": t.tags, "checks": t.check_specs}
                for t in self.tasks
            ],
        }
        return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100)
