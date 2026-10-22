"""Does the gate work? Measure it instead of asserting it.

For each replication we draw a fresh baseline run, then compare it against:
- an unchanged build re-run with a new seed (A/A): the gate must stay quiet,
- the same agent with a planted bug of known strength: the gate should fire.

Over many replications this gives, per gate rule, a false-positive rate and a detection rate
(power) with Wilson intervals. Three gates are compared on identical data:

- naive:     fail if the candidate's mean pass rate is below the baseline's at all
- threshold: fail if the drop is at least 5 points (a common hand-tuned rule)
- agentgate: the significance-tested gate in `gate.py` (quality, cost and critical rules)

Run: python -m agentgate.experiments.seeded_bugs --reps 200 --out results/seeded_bugs.json
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from agentgate.gate import GateConfig, compare
from agentgate.runner import run_suite
from agentgate.sim import BUGS, build_suite, make_agent, make_broken_agent
from agentgate.stats import wilson

RATES = (0.15, 0.35, 0.70)
BROKEN = (("broken_1_task", 1.0), ("broken_3_tasks", 3.0))  # (name, number of tasks broken outright)


def configs() -> list[tuple[str | None, float]]:
    return [(None, 0.0)] + [(b, r) for b in BUGS for r in RATES] + list(BROKEN)


def _agent(bug: str | None, rate: float):
    if bug and bug.startswith("broken_"):
        return make_broken_agent(int(rate))
    return make_agent(bug, rate)


def _rep(args: tuple[int, int, list[tuple[str | None, float]]]) -> list[dict]:
    rep, trials, cfgs = args
    suite = build_suite()
    gate_cfg = GateConfig(n_perm=2000, seed=rep)
    base = run_suite(suite, make_agent(), trials=trials, seed=1_000_003 * rep + 1)
    rows = []
    for i, (bug, rate) in enumerate(cfgs):
        cand = run_suite(suite, _agent(bug, rate), trials=trials, seed=1_000_003 * rep + 2 + i)
        rep_ = compare(base, cand, gate_cfg)
        drop = base.pass_rate - cand.pass_rate
        rows.append({
            "bug": bug or "none", "rate": rate, "drop": drop,
            "naive": drop > 0,
            "threshold": drop >= 0.05,
            "agentgate": rep_.verdict == "fail",
            "by_quality": rep_.quality.failed,
            "by_cost": rep_.steps.failed or rep_.tokens.failed,
            "by_critical": rep_.critical_cand > rep_.critical_base,
            "by_task": bool(rep_.task_regressions),
        })
        rows[-1]["agentgate_core"] = rows[-1]["by_quality"] or rows[-1]["by_cost"] or rows[-1]["by_critical"]
    return rows


def run_experiment(reps: int, trials: int, workers: int, cfgs: list[tuple[str | None, float]] | None = None) -> list[dict]:
    jobs = [(rep, trials, cfgs or configs()) for rep in range(reps)]
    rows: list[dict] = []
    if workers <= 1:
        for j in jobs:
            rows += _rep(j)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for chunk in pool.map(_rep, jobs):
                rows += chunk
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    keys = sorted({(r["bug"], r["rate"]) for r in rows}, key=lambda k: (k[0] != "none", k[0], k[1]))
    out = []
    for bug, rate in keys:
        sel = [r for r in rows if r["bug"] == bug and r["rate"] == rate]
        n = len(sel)
        entry = {"bug": bug, "rate": rate, "n": n, "mean_drop": sum(r["drop"] for r in sel) / n}
        for gate in ("naive", "threshold", "agentgate_core", "agentgate", "by_quality", "by_cost", "by_critical", "by_task"):
            k = sum(r[gate] for r in sel)
            lo, hi = wilson(k, n)
            entry[gate] = {"rate": k / n, "lo": lo, "hi": hi}
        out.append(entry)
    return out


def markdown(summary: list[dict], reps: int, trials: int) -> str:
    f = lambda e, g: f"{e[g]['rate']:.0%}"  # noqa: E731
    aa = next(e for e in summary if e["bug"] == "none")
    lines = [
        f"{reps} replications, {trials} trials per task, 24 tasks. Rates are the share of replications in which the gate fired.",
        "",
        "**Unchanged build (A/A): false-positive rate**",
        "",
        "| gate | fires on an unchanged build | 95% CI |",
        "|---|---|---|",
    ]
    for g, label in (
        ("naive", "naive (any drop)"), ("threshold", "5-point threshold"),
        ("agentgate_core", "agentgate: quality + cost + critical"), ("agentgate", "agentgate: all four rules"),
    ):
        lines.append(f"| {label} | {f(aa, g)} | {aa[g]['lo']:.1%} to {aa[g]['hi']:.1%} |")
    lines += [
        "",
        "**Planted bugs: detection rate**",
        "",
        "| bug | strength | mean pass-rate drop | naive | 5-pt threshold | agentgate (3 rules) | agentgate (all 4) | fired via quality / cost / critical / task |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in summary:
        if e["bug"] == "none":
            continue
        strength = f"{int(e['rate'])} task(s)" if e["bug"].startswith("broken_") else f"{e['rate']:g}"
        lines.append(
            f"| {e['bug']} | {strength} | {e['mean_drop']:.1%} | {f(e, 'naive')} | {f(e, 'threshold')} | "
            f"{f(e, 'agentgate_core')} | {f(e, 'agentgate')} | "
            f"{f(e, 'by_quality')} / {f(e, 'by_cost')} / {f(e, 'by_critical')} / {f(e, 'by_task')} |"
        )
    return "\n".join(lines) + "\n"


def sensitivity(reps: int, workers: int, out: Path) -> str:
    """How do the gates behave as the number of trials per task changes?"""
    cfgs = [(None, 0.0), ("wrong_refund_amount", 0.35)]
    lines = [
        f"{reps} replications per row; A/A = unchanged build; the bug is `wrong_refund_amount` at strength 0.35.",
        "",
        "| trials per task | 5-pt threshold: fires on unchanged build | agentgate: fires on unchanged build | 5-pt threshold: detects bug | agentgate: detects bug |",
        "|---|---|---|---|---|",
    ]
    data = {}
    for k in (3, 5, 10, 20):
        summ = {(e["bug"]): e for e in summarise(run_experiment(reps, k, workers, cfgs))}
        aa, bug = summ["none"], summ["wrong_refund_amount"]
        data[k] = summ
        lines.append(
            f"| {k} | {aa['threshold']['rate']:.0%} | {aa['agentgate']['rate']:.0%} | "
            f"{bug['threshold']['rate']:.0%} | {bug['agentgate']['rate']:.0%} |"
        )
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    table = "\n".join(lines) + "\n"
    out.with_suffix(".md").write_text(table, encoding="utf-8")
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensitivity", action="store_true", help="vary trials per task instead of the main experiment")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("results/seeded_bugs.json"))
    args = ap.parse_args()

    start = time.perf_counter()
    if args.sensitivity:
        print(sensitivity(args.reps, args.workers, args.out))
        print(f"done in {time.perf_counter() - start:.0f}s")
        return
    rows = run_experiment(args.reps, args.trials, args.workers)
    summary = summarise(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"reps": args.reps, "trials": args.trials, "summary": summary}, indent=1), encoding="utf-8")
    md = markdown(summary, args.reps, args.trials)
    args.out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"done in {time.perf_counter() - start:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
