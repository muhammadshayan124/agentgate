"""Is this judge good enough to trust? Measure it against labelled examples first.

Reported: accuracy and Cohen's kappa (agreement beyond chance) with bootstrap intervals, the
confusion matrix, and the two error rates that matter for a *gate*: false accepts (judge
passes a wrong answer, so regressions slip through) and false rejects (judge fails a good
answer, so the gate cries wolf).

A verbosity probe checks a well-known judge bias: padding a wrong answer with confident,
fact-free filler should not make it look right.

`usable` is true only when the lower confidence bound of kappa clears `min_kappa` AND there
are at least `min_n` scored examples. The second rule matters: a bootstrap interval collapses
to a point when agreement is perfect, so 10 right answers out of 10 would otherwise look
certain. A lucky small sample must not be able to certify a judge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agentgate.judge import Example, Judge, JudgeError
from agentgate.stats import cohen_kappa

FLUFF = " I have carefully double-checked every detail and I am completely confident this is accurate and complete."


@dataclass
class CalibrationReport:
    judge: str
    n: int
    abstained: int
    accuracy: float
    accuracy_ci: tuple[float, float]
    kappa: float
    kappa_ci: tuple[float, float]
    tp: int
    fp: int
    fn: int
    tn: int
    false_accept_rate: float  # P(judge says correct | answer actually wrong)
    false_reject_rate: float  # P(judge says wrong | answer actually correct)
    verbosity_delta: float | None  # change in false-accept rate when wrong answers are padded
    min_kappa: float
    min_n: int
    usable: bool
    source_counts: dict[str, int]

    @property
    def human_labelled(self) -> bool:
        return self.source_counts.get("human", 0) == self.n + self.abstained


def load_examples(path: str | Path) -> list[Example]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [Example(**r) for r in rows]


def _ask(judge: Judge, ex: Example) -> bool | None:
    try:
        return judge.judge(ex).correct
    except JudgeError:
        return None


def calibrate(
    judge: Judge,
    examples: list[Example],
    min_kappa: float = 0.6,
    min_n: int = 50,
    n_boot: int = 2000,
    seed: int = 0,
    probe_verbosity: bool = True,
) -> CalibrationReport:
    preds = [(ex, _ask(judge, ex)) for ex in examples]
    answered = [(ex, p) for ex, p in preds if p is not None]
    if len(answered) < 10:
        raise ValueError(f"only {len(answered)} usable verdicts; need at least 10 to calibrate")
    truth = [ex.label for ex, _ in answered]
    pred = [p for _, p in answered]
    n = len(answered)

    tp = sum(t and p for t, p in zip(truth, pred))
    fp = sum((not t) and p for t, p in zip(truth, pred))
    fn = sum(t and (not p) for t, p in zip(truth, pred))
    tn = sum((not t) and (not p) for t, p in zip(truth, pred))
    acc = (tp + tn) / n
    kappa = cohen_kappa(pred, truth)

    rng = np.random.default_rng(seed)
    t_arr, p_arr = np.array(truth), np.array(pred)
    accs, kappas = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        accs.append(float((t_arr[idx] == p_arr[idx]).mean()))
        kappas.append(cohen_kappa(p_arr[idx].tolist(), t_arr[idx].tolist()))
    acc_ci = (float(np.quantile(accs, 0.025)), float(np.quantile(accs, 0.975)))
    kappa_ci = (float(np.quantile(kappas, 0.025)), float(np.quantile(kappas, 0.975)))

    verbosity = None
    if probe_verbosity:
        wrong = [ex for ex, _ in answered if not ex.label]
        if wrong:
            base_fa = sum(bool(_ask(judge, ex)) for ex in wrong) / len(wrong)
            padded_fa = sum(bool(_ask(judge, Example(**{**ex.__dict__, "answer": ex.answer + FLUFF}))) for ex in wrong) / len(wrong)
            verbosity = padded_fa - base_fa

    sources: dict[str, int] = {}
    for ex in examples:
        sources[ex.source] = sources.get(ex.source, 0) + 1

    return CalibrationReport(
        judge=judge.name, n=n, abstained=len(preds) - n, accuracy=acc, accuracy_ci=acc_ci,
        kappa=kappa, kappa_ci=kappa_ci, tp=tp, fp=fp, fn=fn, tn=tn,
        false_accept_rate=fp / (fp + tn) if fp + tn else 0.0,
        false_reject_rate=fn / (fn + tp) if fn + tp else 0.0,
        verbosity_delta=verbosity, min_kappa=min_kappa, min_n=min_n, usable=kappa_ci[0] >= min_kappa and n >= min_n,
        source_counts=sources,
    )


def calibration_markdown(r: CalibrationReport) -> str:
    verdict = "USABLE as a gate input" if r.usable else "NOT usable as a gate input"
    lines = [
        f"### Judge calibration: `{r.judge}` - {verdict}",
        "",
        f"- examples scored: {r.n} (abstained: {r.abstained}); labels: {r.source_counts}",
        f"- accuracy {r.accuracy:.1%} (95% CI {r.accuracy_ci[0]:.1%} to {r.accuracy_ci[1]:.1%})",
        f"- Cohen's kappa {r.kappa:.2f} (95% CI {r.kappa_ci[0]:.2f} to {r.kappa_ci[1]:.2f}); required lower bound >= {r.min_kappa}, and at least {r.min_n} examples",
        f"- false accepts (wrong answer passed): {r.false_accept_rate:.1%}; false rejects (good answer failed): {r.false_reject_rate:.1%}",
        f"- confusion: TP={r.tp} FP={r.fp} FN={r.fn} TN={r.tn}",
    ]
    if r.verbosity_delta is not None:
        lines.append(f"- verbosity probe: padding wrong answers with confident filler changes the false-accept rate by {r.verbosity_delta:+.1%}")
    if not r.human_labelled:
        lines.append("- NOTE: labels are not (all) human-made; see the dataset's `source` field.")
    return "\n".join(lines) + "\n"
