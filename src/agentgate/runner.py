"""Run an agent over a suite, several trials per task, and record what happened.

Trials matter: an agent that passes a task 8 times out of 10 is not "passing" or "failing".
Everything downstream (the gate, confidence intervals) works on per-task pass *rates*.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from agentgate.suite import Suite, Task
from agentgate.trajectory import Trajectory


class Agent(Protocol):
    name: str

    def run(self, task: Task, seed: int) -> Trajectory: ...


def trial_seed(base_seed: int, task_id: str, trial: int) -> int:
    """Stable across processes and platforms (unlike hash())."""
    digest = hashlib.sha256(f"{base_seed}|{task_id}|{trial}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass
class TrialResult:
    task_id: str
    trial: int
    passed: bool
    steps: int
    tokens: int
    failed_checks: list[str] = field(default_factory=list)
    critical_failed: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)


@dataclass
class RunResult:
    suite: str
    agent: str
    trials: int
    seed: int
    results: list[TrialResult]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def task_ids(self) -> list[str]:
        return sorted({r.task_id for r in self.results})

    def by_task(self) -> dict[str, list[TrialResult]]:
        out: dict[str, list[TrialResult]] = {}
        for r in self.results:
            out.setdefault(r.task_id, []).append(r)
        return out

    def task_rates(self) -> dict[str, float]:
        return {t: sum(r.passed for r in rs) / len(rs) for t, rs in self.by_task().items()}

    def task_mean(self, attr: str) -> dict[str, float]:
        return {t: sum(getattr(r, attr) for r in rs) / len(rs) for t, rs in self.by_task().items()}

    @property
    def pass_rate(self) -> float:
        rates = self.task_rates()
        return sum(rates.values()) / len(rates) if rates else 0.0

    @property
    def critical_failures(self) -> int:
        return sum(1 for r in self.results if r.critical_failed)

    def check_failure_counts(self) -> Counter[str]:
        c: Counter[str] = Counter()
        for r in self.results:
            c.update(r.failed_checks)
        return c

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RunResult:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d["results"] = [TrialResult(**r) for r in d["results"]]
        return cls(**d)


def evaluate_trajectory(task: Task, traj: Trajectory) -> TrialResult:
    results = [c.evaluate(traj) for c in task.checks]
    failed = [r for r in results if not r.passed]
    return TrialResult(
        task_id=task.id,
        trial=traj.trial,
        passed=not failed,
        steps=traj.steps,
        tokens=traj.tokens,
        failed_checks=[r.name for r in failed],
        critical_failed=[r.name for r in failed if r.critical],
        reasons={r.name: r.reason for r in failed},
    )


def run_suite(
    suite: Suite,
    agent: Agent,
    trials: int = 10,
    seed: int = 0,
    keep_trajectories: list[Trajectory] | None = None,
) -> RunResult:
    results: list[TrialResult] = []
    for task in suite.tasks:
        for trial in range(trials):
            traj = agent.run(task, trial_seed(seed, task.id, trial))
            traj.task_id, traj.trial = task.id, trial
            results.append(evaluate_trajectory(task, traj))
            if keep_trajectories is not None:
                keep_trajectories.append(traj)
    return RunResult(suite=suite.name, agent=getattr(agent, "name", type(agent).__name__), trials=trials, seed=seed, results=results)


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def load_agent(spec: str, **kwargs: str) -> Agent:
    """`package.module:factory` -> factory(**kwargs). Values from the CLI arrive as strings."""
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise ValueError("agent spec must look like 'package.module:factory'")
    factory = getattr(importlib.import_module(module_name), attr)
    return factory(**{k: _coerce(v) for k, v in kwargs.items()})


def save_trajectories(path: str | Path, trajectories: list[Trajectory]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(json.dumps(t.to_dict()) + "\n")
