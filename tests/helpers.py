"""Build synthetic RunResults with exact per-task pass counts, so gate behaviour can be tested
without running an agent."""

from __future__ import annotations

from agentgate.runner import RunResult, TrialResult


def make_run(
    passes: dict[str, int],
    trials: int = 10,
    steps: int = 4,
    tokens: int = 300,
    critical: dict[str, int] | None = None,
    agent: str = "synthetic",
) -> RunResult:
    critical = critical or {}
    results = []
    for task, k in passes.items():
        for i in range(trials):
            ok = i < k
            crit = i < critical.get(task, 0)
            results.append(TrialResult(
                task_id=task, trial=i, passed=ok and not crit, steps=steps, tokens=tokens,
                failed_checks=[] if ok and not crit else ["some_check"],
                critical_failed=["never_do_x"] if crit else [],
            ))
    return RunResult(suite="s", agent=agent, trials=trials, seed=0, results=results)


def uniform(n_tasks: int, k: int) -> dict[str, int]:
    return {f"t{i:02d}": k for i in range(n_tasks)}
