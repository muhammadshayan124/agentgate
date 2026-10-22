"""Human-readable output for the gate: terminal text and a Markdown block for PR comments."""

from __future__ import annotations

from agentgate.gate import GateReport, MetricComparison
from agentgate.mde import PowerPoint
from agentgate.runner import RunResult


def _row(m: MetricComparison, pct: bool) -> str:
    fmt = (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v:.1f}")
    change = f"{m.diff:+.1%}" if pct else (m.note or f"{m.diff:+.1f}")
    flag = "FAIL" if m.failed else "ok"
    return f"| {m.name} | {fmt(m.base)} | {fmt(m.cand)} | {change} | {m.ci_lo:+.3f} to {m.ci_hi:+.3f} | {m.p_value:.4f} | {flag} |"


def gate_markdown(rep: GateReport) -> str:
    icon = "PASS" if rep.passed else "FAIL"
    lines = [
        f"### agentgate: {icon}",
        "",
        f"{rep.n_tasks} tasks x {rep.trials} trials per run. `diff` is the paired per-task change "
        "(positive = candidate worse); the interval is a 95% bootstrap over tasks.",
        "",
        "| metric | baseline | candidate | change | 95% interval (diff) | p | status |",
        "|---|---|---|---|---|---|---|",
        _row(rep.quality, True),
        _row(rep.steps, False),
        _row(rep.tokens, False),
        f"| critical-check failures (trials) | {rep.critical_base} | {rep.critical_cand} | | | | "
        f"{'FAIL' if rep.critical_cand > rep.critical_base else 'ok'} |",
        f"| individually regressed tasks | | {len(rep.task_regressions)} | | | | {'FAIL' if rep.task_regressions else 'ok'} |",
    ]
    if rep.reasons:
        lines += ["", "**Why it failed**", ""] + [f"- {r}" for r in rep.reasons]
    if rep.worst_tasks:
        lines += ["", "**Most regressed tasks**", "", "| task | baseline | candidate |", "|---|---|---|"]
        lines += [f"| {t} | {b:.0%} | {c:.0%} |" for t, b, c in rep.worst_tasks]
    if rep.worst_checks:
        lines += ["", "**Checks failing more often**", "", "| check | baseline fails | candidate fails |", "|---|---|---|"]
        lines += [f"| `{n}` | {b} | {c} |" for n, b, c in rep.worst_checks]
    return "\n".join(lines) + "\n"


def run_summary(run: RunResult) -> str:
    lines = [
        f"{run.agent}: {run.pass_rate:.1%} mean task pass rate over {len(run.task_ids)} tasks x {run.trials} trials; "
        f"{run.critical_failures} trial(s) with a critical failure",
    ]
    for name, n in run.check_failure_counts().most_common(5):
        lines.append(f"  {n:4d} x {name}")
    return "\n".join(lines)


def power_markdown(points: list[PowerPoint], shape: str, mde: float | None) -> str:
    lines = [f"Power of the gate vs the size of a **{shape}** regression:", "", "| mean pass-rate drop | power |", "|---|---|"]
    lines += [f"| {p.drop:.1%} | {p.power:.0%} |" for p in points]
    lines.append("")
    lines.append(f"Smallest drop detected with >= 80% power: {mde:.1%}" if mde is not None else "No tested drop reached 80% power.")
    return "\n".join(lines)
