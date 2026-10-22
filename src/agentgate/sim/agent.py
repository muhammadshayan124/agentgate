"""A deterministic stand-in for an LLM agent, used to evaluate the evaluator.

Why simulate? To know whether a regression gate works you must be able to (a) run the same
agent twice and see the gate stay quiet, and (b) plant a bug of known size and see it fire.
Real models cannot give you either on demand, free and reproducibly. The harness itself is
agent-agnostic; this module is only the controlled test subject.

What it models honestly:
- Flakiness. Even the "good" agent slips occasionally (forgets the email, a typo in an amount,
  a redundant lookup, a bad search query), and some tasks are harder than others. Without
  this, every run would be identical and statistics would be pointless.
- Real tool execution. Decisions depend on actual tool results from `env.py`.

What it does not model: language, reasoning, or any real model behaviour. Do not read the
numbers from this subject as a statement about LLMs.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field

from agentgate.sim.env import ORDERS, Env
from agentgate.suite import Task
from agentgate.trajectory import ToolCall, Trajectory

# Benign slips of the baseline agent: (probability per trial, intents it applies to).
BASE_NOISE: dict[str, tuple[float, tuple[str, ...]]] = {
    "forget_email": (0.03, ("refund",)),
    "typo_amount": (0.02, ("refund",)),
    "redundant_lookup": (0.04, ("refund", "status", "kb")),  # a retried read, never a repeated write
    "wrong_kb_query": (0.10, ("kb",)),
    "terse_answer": (0.02, ("refund", "status", "kb", "legal")),
}

# Injectable regressions: name -> (description, intents it can affect).
BUGS: dict[str, tuple[str, tuple[str, ...]]] = {
    "skip_policy_check": ("never calls check_refund_policy and assumes the refund is allowed", ("refund",)),
    "wrong_refund_amount": ("refunds 10% too much", ("refund",)),
    "refund_forbidden_orders": ("calls the policy tool but refunds even when it says no", ("refund",)),
    "no_confirmation_email": ("never sends the confirmation email", ("refund",)),
    "redundant_calls": ("repeats its last tool call", ("refund", "status", "kb", "legal")),
    "runaway_loop": ("re-reads the order eight extra times (cost blow-up)", ("refund", "status", "legal")),
    "bad_kb_query": ("searches the knowledge base for the wrong thing", ("kb",)),
    "hallucinate_status": ("states the wrong order status", ("status",)),
    "missed_escalation": ("refunds an angry customer instead of escalating", ("legal",)),
}


def _difficulty(task_id: str) -> float:
    """Stable per-task multiplier in [0.5, 2.0]: some tasks are just harder."""
    return 0.5 + 1.5 * hashlib.sha256(task_id.encode()).digest()[0] / 255


@dataclass
class SimAgent:
    bug: str | None = None
    rate: float = 0.0
    noise: float = 1.0

    def __post_init__(self) -> None:
        if self.bug is not None and self.bug not in BUGS:
            raise ValueError(f"unknown bug '{self.bug}'. Known: {sorted(BUGS)}")

    @property
    def name(self) -> str:
        return "sim-baseline" if not self.bug else f"sim+{self.bug}@{self.rate:g}"

    # -- helpers -----------------------------------------------------------------------
    def _slip(self, rng: random.Random, kind: str, intent: str, difficulty: float) -> bool:
        p, intents = BASE_NOISE[kind]
        return intent in intents and rng.random() < min(1.0, p * difficulty * self.noise)

    def _bug(self, rng: random.Random, bug: str, intent: str) -> bool:
        return self.bug == bug and intent in BUGS[bug][1] and rng.random() < self.rate

    @staticmethod
    def _call(env: Env, traj: Trajectory, name: str, **args):
        result, error = env.call(name, args)
        tokens = 40 + len(json.dumps(args)) // 4 + len(json.dumps(result)) // 4
        traj.calls.append(ToolCall(name=name, args=args, result=result, error=error, tokens=tokens))
        return result

    # -- the policy ----------------------------------------------------------------------
    def run(self, task: Task, seed: int) -> Trajectory:
        rng = random.Random(seed)
        env, traj = Env(), Trajectory(task_id=task.id)
        intent, p = task.params["intent"], task.params
        diff = _difficulty(task.id)
        terse = self._slip(rng, "terse_answer", intent, diff)

        if intent == "refund":
            answer = self._refund(env, traj, p["order_id"], rng, diff)
        elif intent == "status":
            answer = self._status(env, traj, p["order_id"], rng)
        elif intent == "kb":
            answer = self._kb(env, traj, p["kb_query"], rng, diff)
        elif intent == "legal":
            answer = self._legal(env, traj, p["order_id"], rng)
        else:
            raise ValueError(f"unknown intent {intent}")

        if self._slip(rng, "redundant_lookup", intent, diff):
            first = traj.calls[0]  # always a read (lookup_order / search_kb)
            self._call(env, traj, first.name, **first.args)
        if self._bug(rng, "redundant_calls", intent):
            last = traj.calls[-1]  # may be a write: a retry that double-refunds
            self._call(env, traj, last.name, **last.args)
        if self._bug(rng, "runaway_loop", intent):
            for _ in range(8):
                self._call(env, traj, "lookup_order", order_id=p["order_id"])

        traj.answer = "Done." if terse else answer
        traj.answer_tokens = int(len(traj.answer.split()) * 1.3) + 5
        return traj

    def _refund(self, env: Env, traj: Trajectory, oid: str, rng: random.Random, diff: float) -> str:
        order = self._call(env, traj, "lookup_order", order_id=oid)
        if self._bug(rng, "skip_policy_check", "refund"):
            eligible, reason = True, "assumed"
        else:
            policy = self._call(env, traj, "check_refund_policy", order_id=oid)
            eligible, reason = policy["eligible"], policy["reason"]
        if not eligible and self._bug(rng, "refund_forbidden_orders", "refund"):
            eligible = True

        send_email = not (self._slip(rng, "forget_email", "refund", diff) or self._bug(rng, "no_confirmation_email", "refund"))
        email = order["customer_email"]
        if eligible:
            amount = order["total"]
            if self._slip(rng, "typo_amount", "refund", diff):
                amount = round(amount + 0.10, 2)
            if self._bug(rng, "wrong_refund_amount", "refund"):
                amount = round(amount * 1.10, 2)
            self._call(env, traj, "issue_refund", order_id=oid, amount=amount)
            if send_email:
                self._call(env, traj, "send_email", to=email, subject="Your refund", body=f"Refunded ${amount:.2f}.")
            return f"I've issued a refund of ${amount:.2f} for order {oid} and emailed you a confirmation."
        if send_email:
            self._call(env, traj, "send_email", to=email, subject="About your refund request", body=reason)
        return f"Order {oid} is not eligible for a refund because {reason}."

    def _status(self, env: Env, traj: Trajectory, oid: str, rng: random.Random) -> str:
        order = self._call(env, traj, "lookup_order", order_id=oid)
        status = order["status"]
        if self._bug(rng, "hallucinate_status", "status"):
            status = rng.choice([s for s in ("shipped", "processing", "delivered") if s != status])
        eta = f" It should arrive in {order['eta']}." if order["eta"] and status != "delivered" else ""
        return f"Order {oid} is {status}.{eta}"

    def _kb(self, env: Env, traj: Trajectory, key: str, rng: random.Random, diff: float) -> str:
        query = "weather today" if (self._slip(rng, "wrong_kb_query", "kb", diff) or self._bug(rng, "bad_kb_query", "kb")) else key
        result = self._call(env, traj, "search_kb", query=query)
        return result["answer"] or "I'm not sure about that; let me find out and get back to you."

    def _legal(self, env: Env, traj: Trajectory, oid: str, rng: random.Random) -> str:
        if self._bug(rng, "missed_escalation", "legal"):
            amount = ORDERS[oid].total
            self._call(env, traj, "lookup_order", order_id=oid)
            self._call(env, traj, "issue_refund", order_id=oid, amount=amount)
            return f"I've issued a refund of ${amount:.2f} for order {oid}."
        self._call(env, traj, "escalate", reason=f"Customer threatening legal action over order {oid}")
        return "I've escalated this to a specialist who will contact you shortly."


def make_agent(bug: str | None = None, rate: float = 0.0, noise: float = 1.0) -> SimAgent:
    return SimAgent(bug=bug or None, rate=float(rate), noise=float(noise))


# Tasks that `BrokenTaskAgent` can break outright: one refund, one status query, one KB question.
BREAKABLE = ("refund-ok-A1003", "status-C3002", "kb-2")


@dataclass
class BrokenTaskAgent:
    """The baseline agent, except that `n` specific tasks fail completely (it does nothing).

    Models a regression confined to one code path: a mean over all tasks barely moves, but those
    tasks go from ~90% to 0%.
    """

    n: int = 1
    inner: SimAgent = field(default_factory=SimAgent)

    @property
    def name(self) -> str:
        return f"sim+broken_{self.n}_task{'s' if self.n != 1 else ''}"

    def run(self, task: Task, seed: int) -> Trajectory:
        if task.id in BREAKABLE[: self.n]:
            return Trajectory(task_id=task.id, answer="Sorry, something went wrong.", answer_tokens=6)
        return self.inner.run(task, seed)


def make_broken_agent(n: int = 1) -> BrokenTaskAgent:
    return BrokenTaskAgent(n=int(n))
