"""`agentgate demo`: the gate dogfooded on the simulated agent.

An unchanged build must pass; each planted bug must fail. `--check` turns that into an exit
code, so CI runs it on every commit as an end-to-end test of the gate itself.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from agentgate.gate import compare
from agentgate.runner import run_suite
from agentgate.sim import build_suite, make_agent, make_broken_agent

# (bug, rate): strong enough that a sound gate must fire on them, at the default settings.
PLANTED = [
    ("skip_policy_check", 0.5),
    ("wrong_refund_amount", 0.5),
    ("refund_forbidden_orders", 0.5),
    ("runaway_loop", 0.5),
    ("hallucinate_status", 0.5),
    ("missed_escalation", 0.7),
]


def run_demo(console: Console, trials: int = 10) -> bool:
    suite = build_suite()
    baseline = run_suite(suite, make_agent(), trials=trials, seed=1)
    table = Table(title=f"agentgate demo: {len(suite.tasks)} tasks x {trials} trials, baseline pass rate {baseline.pass_rate:.1%}")
    table.add_column("candidate", no_wrap=True)
    for col in ("pass", "change", "p", "crit", "verdict"):
        table.add_column(col, no_wrap=True)

    ok = True
    cases = [("same build, new seed", make_agent(), "pass")] + [
        (f"{b[:20]}@{r:g}", make_agent(b, r), "fail") for b, r in PLANTED
    ] + [("ONE task fully broken", make_broken_agent(1), "fail")]
    for label, agent, expected in cases:
        cand = run_suite(suite, agent, trials=trials, seed=2)
        rep = compare(baseline, cand)
        good = rep.verdict == expected
        ok &= good
        table.add_row(
            label, f"{rep.quality.cand:.1%}", f"{-rep.quality.diff:+.1%}", f"{rep.quality.p_value:.4f}",
            f"{rep.critical_cand}", f"{rep.verdict.upper()} {'(ok)' if good else '(UNEXPECTED)'}",
        )
    console.print(table)
    console.print("gate behaved as expected on every case" if ok else "UNEXPECTED gate behaviour")
    return ok
