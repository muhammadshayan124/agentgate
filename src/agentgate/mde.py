"""How small a regression can this suite actually detect?

A gate that never fires is useless; one that fires on noise is worse. Before trusting a suite
you should know its power: the probability the gate catches a regression of a given size.
This simulates it from a baseline run's own per-task pass rates (no model needed).

Two regression shapes are simulated, because they behave differently:
- `uniform`: every task gets slightly worse (a diffuse quality loss).
- `concentrated`: a few tasks break completely (a bug in one code path). For the same mean
  drop this is HARDER for the mean-based quality rule: large swings on a few tasks inflate the
  null variance. The per-task rule in `gate.py` exists to catch that case; this simulation
  covers the quality rule only, so it is the pessimistic view for concentrated regressions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agentgate.runner import RunResult
from agentgate.stats import signflip_pvalue


@dataclass
class PowerPoint:
    drop: float
    power: float


def _simulate_gate(
    base_rates: np.ndarray, cand_rates: np.ndarray, trials: int, alpha: float, rng: np.random.Generator, n_perm: int
) -> bool:
    b = rng.binomial(trials, base_rates) / trials
    c = rng.binomial(trials, cand_rates) / trials
    d = b - c
    return bool(signflip_pvalue(d, rng, n_perm) < alpha and d.mean() > 0)


def power_curve(
    base: RunResult,
    drops: list[float],
    shape: str = "uniform",
    alpha: float = 0.05,
    sims: int = 300,
    n_perm: int = 2_000,
    seed: int = 0,
) -> list[PowerPoint]:
    rng = np.random.default_rng(seed)
    rates = np.array(list(base.task_rates().values()))
    out = []
    for drop in drops:
        if shape == "uniform":
            cand = np.clip(rates - drop, 0.0, 1.0)
        elif shape == "concentrated":
            # Break the fewest tasks (lowest-loss first) needed to reach the target mean drop.
            cand = rates.copy()
            need = drop * rates.size
            for i in np.argsort(-rates):
                if need <= 1e-12:
                    break
                take = min(cand[i], need)
                cand[i] -= take
                need -= take
        else:
            raise ValueError("shape must be 'uniform' or 'concentrated'")
        hits = sum(_simulate_gate(rates, cand, base.trials, alpha, rng, n_perm) for _ in range(sims))
        out.append(PowerPoint(drop, hits / sims))
    return out


def minimum_detectable_drop(points: list[PowerPoint], target_power: float = 0.8) -> float | None:
    for p in sorted(points, key=lambda x: x.drop):
        if p.power >= target_power:
            return p.drop
    return None
