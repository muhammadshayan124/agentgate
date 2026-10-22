"""The regression gate: compare a candidate run against a baseline run and decide.

Four independent rules, so a failure always says why:

1. Quality: the paired per-task drop in pass rate must be statistically significant
   (one-sided sign-flip test, `alpha`) and at least `min_effect` to fail the build.
2. Cost: steps and tokens may not rise by more than `max_cost_increase` (relative), and the
   rise must be significant. Without this an agent can "pass" by becoming far more expensive.
3. Critical: a "must never happen" check may not fail more often than in the baseline.
   No statistics here by design: one new safety violation is enough.
4. Task: any single task whose pass rate fell by at least `task_min_drop` and is significant
   under an exact test with Holm correction across all tasks. Rule 1 averages over tasks, so it
   is nearly blind to ONE task breaking completely (100% -> 0% gives p = 0.5 on 24 tasks);
   this rule exists for that case.

Why not just compare the two mean pass rates? Because agents are stochastic. Two runs of the
*same* agent differ by chance, so "candidate < baseline" fires on a large share of identical
builds (see the experiment in the README). The significance test is what keeps the gate quiet
when nothing changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from agentgate.runner import RunResult
from agentgate.stats import bootstrap_mean_ci, fisher_lower_pvalue, holm_reject, signflip_pvalue


@dataclass
class GateConfig:
    alpha: float = 0.05
    min_effect: float = 0.0  # absolute drop in mean pass rate that is worth failing for
    max_cost_increase: float = 0.20  # relative rise in mean steps / tokens
    critical_allowance: int = 0  # extra critical-failure trials tolerated over the baseline
    task_rule: bool = True
    task_min_drop: float = 0.5  # a single task must fall by at least this much to count
    n_perm: int = 10_000
    seed: int = 0


@dataclass
class MetricComparison:
    name: str
    base: float
    cand: float
    diff: float  # positive = worse (pass-rate drop, or cost increase)
    ci_lo: float
    ci_hi: float
    p_value: float
    failed: bool
    note: str = ""


@dataclass
class GateReport:
    verdict: str
    reasons: list[str]
    quality: MetricComparison
    steps: MetricComparison
    tokens: MetricComparison
    critical_base: int
    critical_cand: int
    n_tasks: int
    trials: int
    task_regressions: list[tuple[str, float, float, float]] = field(default_factory=list)  # (task, base, cand, p)
    worst_tasks: list[tuple[str, float, float]] = field(default_factory=list)  # (task, base, cand)
    worst_checks: list[tuple[str, int, int]] = field(default_factory=list)  # (check, base fails, cand fails)

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


def _align(base: RunResult, cand: RunResult) -> list[str]:
    common = sorted(set(base.task_ids) & set(cand.task_ids))
    if not common:
        raise ValueError("baseline and candidate share no tasks; were they run on the same suite?")
    return common


def compare(base: RunResult, cand: RunResult, cfg: GateConfig | None = None) -> GateReport:
    cfg = cfg or GateConfig()
    rng = np.random.default_rng(cfg.seed)
    tasks = _align(base, cand)

    br, cr = base.task_rates(), cand.task_rates()
    drop = np.array([br[t] - cr[t] for t in tasks])  # positive = candidate worse
    mean_drop = float(drop.mean())
    p_q = signflip_pvalue(drop, rng, cfg.n_perm)
    lo, hi = bootstrap_mean_ci(drop, rng, cfg.alpha)
    q_failed = p_q < cfg.alpha and mean_drop >= max(cfg.min_effect, 1e-12)
    quality = MetricComparison(
        "pass rate", float(np.mean([br[t] for t in tasks])), float(np.mean([cr[t] for t in tasks])),
        mean_drop, lo, hi, p_q, q_failed,
    )

    def cost(attr: str, label: str) -> MetricComparison:
        bm, cm = base.task_mean(attr), cand.task_mean(attr)
        inc = np.array([cm[t] - bm[t] for t in tasks])
        b_mean = float(np.mean([bm[t] for t in tasks]))
        c_mean = float(np.mean([cm[t] for t in tasks]))
        rel = (c_mean - b_mean) / b_mean if b_mean else 0.0
        p = signflip_pvalue(inc, rng, cfg.n_perm)
        clo, chi = bootstrap_mean_ci(inc, rng, cfg.alpha)
        failed = p < cfg.alpha and rel > cfg.max_cost_increase
        return MetricComparison(label, b_mean, c_mean, c_mean - b_mean, clo, chi, p, failed, note=f"{rel:+.1%}")

    steps, tokens = cost("steps", "steps"), cost("tokens", "tokens")

    crit_b, crit_c = base.critical_failures, cand.critical_failures
    crit_failed = crit_c > crit_b + cfg.critical_allowance

    task_regs: list[tuple[str, float, float, float]] = []
    if cfg.task_rule:
        by_b, by_c = base.by_task(), cand.by_task()
        pvals = [
            fisher_lower_pvalue(
                sum(r.passed for r in by_b[t]), len(by_b[t]), sum(r.passed for r in by_c[t]), len(by_c[t])
            )
            for t in tasks
        ]
        for t, pv, rej in zip(tasks, pvals, holm_reject(pvals, cfg.alpha)):
            if rej and br[t] - cr[t] >= cfg.task_min_drop:
                task_regs.append((t, br[t], cr[t], pv))
        task_regs.sort(key=lambda x: x[3])

    reasons: list[str] = []
    if quality.failed:
        reasons.append(
            f"pass rate fell {mean_drop:.1%} ({quality.base:.1%} -> {quality.cand:.1%}), "
            f"p={p_q:.4f} < alpha={cfg.alpha}"
        )
    for m in (steps, tokens):
        if m.failed:
            reasons.append(f"{m.name} rose {m.note} ({m.base:.1f} -> {m.cand:.1f}), p={m.p_value:.4f}")
    if crit_failed:
        reasons.append(f"critical checks failed in {crit_c} trial(s) vs {crit_b} in the baseline")
    if task_regs:
        shown = ", ".join(f"{t} ({b:.0%} -> {c:.0%})" for t, b, c, _ in task_regs[:5])
        reasons.append(f"{len(task_regs)} task(s) regressed individually (exact test, Holm-corrected): {shown}")

    deltas = sorted(((t, br[t], cr[t]) for t in tasks), key=lambda x: x[1] - x[2], reverse=True)
    worst_tasks = [d for d in deltas if d[1] - d[2] > 0][:5]
    bc, cc = base.check_failure_counts(), cand.check_failure_counts()
    worst_checks = sorted(
        ((name, bc.get(name, 0), cc.get(name, 0)) for name in set(bc) | set(cc)),
        key=lambda x: x[2] - x[1],
        reverse=True,
    )
    worst_checks = [w for w in worst_checks if w[2] > w[1]][:5]

    return GateReport(
        verdict="fail" if reasons else "pass",
        reasons=reasons,
        quality=quality,
        steps=steps,
        tokens=tokens,
        critical_base=crit_b,
        critical_cand=crit_c,
        n_tasks=len(tasks),
        trials=min(base.trials, cand.trials),
        task_regressions=task_regs,
        worst_tasks=worst_tasks,
        worst_checks=worst_checks,
    )
