"""Labelled examples for judge calibration, generated from the simulated backend.

Each example pairs real tool output (the evidence) with an agent answer that is correct or
wrong *by construction*: wrong answers are built by corrupting a fact (status, amount,
decision, duration) or by negating it. So the labels are exact, but they are NOT human
judgements and the wrong answers are templated, which makes them easier than real model
output. A judge that fails here is unusable; one that passes here has only cleared a low bar.
Examples carry `source: constructed` so reports say so. Use `source: human` for real labels.
"""

from __future__ import annotations

import json
import random

from agentgate.judge import Example
from agentgate.sim.env import KB, ORDERS, Env

STATUSES = ("shipped", "processing", "delivered")

_STATUS_OK = [
    "Order {oid} is {status}.",
    "Good news, your order {oid} currently shows as {status}.",
    "I checked: order {oid} has the status '{status}'.",
]
_STATUS_NEG = {
    "delivered": "Order {oid} has not been delivered yet.",
    "shipped": "Order {oid} has not shipped yet.",
    "processing": "Order {oid} has already been processed and is on its way.",
}
_REFUND_OK = [
    "I've issued a refund of ${amount:.2f} for order {oid}.",
    "Your refund of ${amount:.2f} for order {oid} has been processed and a confirmation email is on its way.",
]
_REFUND_WRONG_AMT = [
    "I've issued a refund of ${amount:.2f} for order {oid}.",
    "Your refund of ${amount:.2f} for order {oid} has been processed.",
]
_NOT_ELIGIBLE_OK = [
    "Order {oid} is not eligible for a refund because {reason}.",
    "I'm sorry, I can't refund order {oid}: {reason}.",
]
_REFUND_WHEN_FORBIDDEN = [
    "I've issued a full refund for order {oid}.",
    "Good news, your refund for order {oid} is approved and on its way.",
]
_NOT_ELIGIBLE_WRONG = [
    "Order {oid} is not eligible for a refund because it is outside the refund window.",
    "I'm sorry, I can't refund order {oid}; refunds are not available for this item.",
]


def _evidence(env: Env, oid: str) -> str:
    order, _ = env.call("lookup_order", {"order_id": oid})
    policy, _ = env.call("check_refund_policy", {"order_id": oid})
    return f"lookup_order -> {json.dumps(order)}\ncheck_refund_policy -> {json.dumps(policy)}"


def build_examples(seed: int = 0, rounds: int = 3) -> list[Example]:
    """`rounds` passes with different template/corruption choices, de-duplicated."""
    out: list[Example] = []
    seen: set[tuple[str, str, str]] = set()
    for r in range(rounds):
        _one_round(random.Random(seed + r), out, seen)
    random.Random(seed).shuffle(out)
    return [Example(**{**e.__dict__, "id": f"{e.kind}-{i:03d}"}) for i, e in enumerate(out)]


def _one_round(rng: random.Random, out: list[Example], seen: set[tuple[str, str, str]]) -> None:
    env = Env()

    def add(kind: str, q: str, ev: str, ans: str, label: bool) -> None:
        key = (q, ev, ans)
        if key not in seen:
            seen.add(key)
            out.append(Example(id="", question=q, evidence=ev, answer=ans, label=label, kind=kind))

    for oid, o in ORDERS.items():
        ev = _evidence(env, oid)
        if oid.startswith("C"):  # status questions
            q = f"Where is my order {oid}?"
            add("status", q, ev, rng.choice(_STATUS_OK).format(oid=oid, status=o.status), True)
            other = rng.choice([s for s in STATUSES if s != o.status])
            add("status", q, ev, rng.choice(_STATUS_OK).format(oid=oid, status=other), False)
            add("status", q, ev, _STATUS_NEG[o.status].format(oid=oid), False)
        elif oid.startswith("A"):  # eligible refunds
            q = f"Please refund order {oid}."
            add("refund_amount", q, ev, rng.choice(_REFUND_OK).format(oid=oid, amount=o.total), True)
            bad = round(o.total * rng.choice([1.1, 0.5, 2.0]), 2)
            add("refund_amount", q, ev, rng.choice(_REFUND_WRONG_AMT).format(oid=oid, amount=bad), False)
            add("eligibility", q, ev, rng.choice(_NOT_ELIGIBLE_WRONG).format(oid=oid), False)
        elif oid.startswith("B"):  # forbidden refunds
            q = f"Please refund order {oid}."
            reason = o.refund_decision()[1]
            add("eligibility", q, ev, rng.choice(_NOT_ELIGIBLE_OK).format(oid=oid, reason=reason), True)
            add("eligibility", q, ev, rng.choice(_REFUND_WHEN_FORBIDDEN).format(oid=oid), False)
            add("refund_amount", q, ev, rng.choice(_REFUND_OK).format(oid=oid, amount=o.total), False)

    for key, text in KB.items():
        ev = f"search_kb -> {json.dumps({'answer': text})}"
        q = f"Question about {key}"
        add("kb", q, ev, text, True)
        add("kb", q, ev, f"Sure! {text}", True)
        wrong = {"shipping": "Standard shipping takes 1-2 business days.",
                 "return window": "You can return most items within 90 days of delivery.",
                 "international": "We only ship within the US."}[key]
        add("kb", q, ev, wrong, False)
        add("kb", q, ev, "I'm not sure about that; let me find out and get back to you.", False)



def write_examples(path: str) -> int:
    ex = build_examples()
    with open(path, "w", encoding="utf-8") as f:
        for e in ex:
            f.write(json.dumps(e.__dict__) + "\n")
    return len(ex)
