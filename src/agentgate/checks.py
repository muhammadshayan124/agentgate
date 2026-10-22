"""Deterministic checks over a trajectory.

Each check returns pass/fail with a reason. A check can be marked `critical`: a "must never
happen" rule (issuing a refund the policy forbids, a double charge). Critical checks are
gated without statistics: one new violation fails the build.

Checks are declared in YAML as flat dicts: `{type: tool_called, tool: lookup_order, ...}`.
String values of the form `${name}` are filled from the task's `params` (type preserved).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentgate.trajectory import Trajectory


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    reason: str
    critical: bool = False


class Check:
    """Base class. Subclasses implement `_evaluate` returning (passed, reason)."""

    type: str = ""

    def __init__(self, name: str | None = None, critical: bool = False):
        self.name = name or self.type
        self.critical = critical

    def _evaluate(self, traj: Trajectory) -> tuple[bool, str]:
        raise NotImplementedError

    def evaluate(self, traj: Trajectory) -> CheckResult:
        try:
            passed, reason = self._evaluate(traj)
        except Exception as exc:  # a broken check must fail loudly, not pass silently
            passed, reason = False, f"check raised {type(exc).__name__}: {exc}"
        return CheckResult(self.name, bool(passed), reason, self.critical)


_REGISTRY: dict[str, Callable[..., Check]] = {}


def register(cls: type[Check]) -> type[Check]:
    _REGISTRY[cls.type] = cls
    return cls


def _values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)) and not isinstance(expected, bool):
        return abs(float(expected) - float(actual)) < 1e-6
    return expected == actual


def args_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    """Every expected key must be present with an equal value (extra actual keys are fine)."""
    return all(k in actual and _values_match(v, actual[k]) for k, v in expected.items())


@register
class ToolCalled(Check):
    type = "tool_called"

    def __init__(self, tool: str, args: dict[str, Any] | None = None, min: int = 1, max: int | None = None, **kw):
        super().__init__(**kw)
        self.tool, self.args, self.min, self.max = tool, args or {}, min, max
        if "name" not in kw:
            self.name = f"tool_called:{tool}"

    def _evaluate(self, traj: Trajectory) -> tuple[bool, str]:
        matching = [c for c in traj.calls if c.name == self.tool and args_match(self.args, c.args)]
        n = len(matching)
        if n < self.min:
            seen = [c.args for c in traj.calls if c.name == self.tool]
            hint = f" (called with {seen})" if seen else ""
            return False, f"expected >= {self.min} call(s) to {self.tool} matching {self.args}, got {n}{hint}"
        if self.max is not None and n > self.max:
            return False, f"expected <= {self.max} call(s) to {self.tool}, got {n}"
        return True, "ok"


@register
class ToolNotCalled(Check):
    type = "tool_not_called"

    def __init__(self, tool: str, **kw):
        super().__init__(**kw)
        self.tool = tool
        if "name" not in kw:
            self.name = f"tool_not_called:{tool}"

    def _evaluate(self, traj: Trajectory) -> tuple[bool, str]:
        n = sum(1 for c in traj.calls if c.name == self.tool)
        return (n == 0, "ok" if n == 0 else f"{self.tool} was called {n} time(s)")


@register
class Order(Check):
    type = "order"

    def __init__(self, tools: list[str], **kw):
        super().__init__(**kw)
        self.tools = list(tools)
        if "name" not in kw:
            self.name = "order:" + ">".join(self.tools)

    def _evaluate(self, traj: Trajectory) -> tuple[bool, str]:
        it = iter(traj.tool_names)
        missing = [t for t in self.tools if t not in it]  # consumes the iterator: subsequence test
        return (not missing, "ok" if not missing else f"calls not in required order; first missing: {missing[0]}")


@register
class MaxSteps(Check):
    type = "max_steps"

    def __init__(self, n: int, **kw):
        super().__init__(**kw)
        self.n = n

    def _evaluate(self, traj: Trajectory) -> tuple[bool, str]:
        return (traj.steps <= self.n, "ok" if traj.steps <= self.n else f"{traj.steps} steps > budget {self.n}")


@register
class MaxTokens(Check):
    type = "max_tokens"

    def __init__(self, n: int, **kw):
        super().__init__(**kw)
        self.n = n

    def _evaluate(self, traj: Trajectory) -> tuple[bool, str]:
        return (traj.tokens <= self.n, "ok" if traj.tokens <= self.n else f"{traj.tokens} tokens > budget {self.n}")


@register
class NoRepeatCalls(Check):
    type = "no_repeat_calls"

    def _evaluate(self, traj: Trajectory) -> tuple[bool, str]:
        seen: set[str] = set()
        for c in traj.calls:
            key = f"{c.name}:{sorted(c.args.items(), key=lambda kv: kv[0])}"
            if key in seen:
                return False, f"identical call repeated: {c.name}({c.args})"
            seen.add(key)
        return True, "ok"


@register
class NoToolErrors(Check):
    type = "no_tool_errors"

    def _evaluate(self, traj: Trajectory) -> tuple[bool, str]:
        bad = [c.name for c in traj.calls if c.error]
        return (not bad, "ok" if not bad else f"tool errors in: {bad}")


@register
class AnswerContains(Check):
    type = "answer_contains"

    def __init__(self, values: list[str] | str, mode: str = "all", **kw):
        super().__init__(**kw)
        self.values = [values] if isinstance(values, str) else list(values)
        self.mode = mode
        if "name" not in kw:
            self.name = "answer_contains:" + "|".join(self.values)

    def _evaluate(self, traj: Trajectory) -> tuple[bool, str]:
        text = traj.answer.lower()
        found = [v for v in self.values if str(v).lower() in text]
        ok = len(found) == len(self.values) if self.mode == "all" else bool(found)
        missing = [v for v in self.values if v not in found]
        return ok, "ok" if ok else f"answer is missing {missing}"


@register
class AnswerNotContains(Check):
    type = "answer_not_contains"

    def __init__(self, values: list[str] | str, **kw):
        super().__init__(**kw)
        self.values = [values] if isinstance(values, str) else list(values)
        if "name" not in kw:
            self.name = "answer_not_contains:" + "|".join(self.values)

    def _evaluate(self, traj: Trajectory) -> tuple[bool, str]:
        text = traj.answer.lower()
        bad = [v for v in self.values if str(v).lower() in text]
        return (not bad, "ok" if not bad else f"answer contains forbidden {bad}")


@register
class AnswerRegex(Check):
    type = "answer_regex"

    def __init__(self, pattern: str, **kw):
        super().__init__(**kw)
        self.pattern = re.compile(pattern, re.IGNORECASE | re.DOTALL)
        if "name" not in kw:
            self.name = f"answer_regex:{pattern}"

    def _evaluate(self, traj: Trajectory) -> tuple[bool, str]:
        ok = bool(self.pattern.search(traj.answer))
        return ok, "ok" if ok else f"answer does not match /{self.pattern.pattern}/"


def substitute(obj: Any, params: dict[str, Any]) -> Any:
    """Fill `${name}` placeholders from params. A string that is exactly one placeholder keeps
    the parameter's type (so numbers stay numbers)."""
    if isinstance(obj, str):
        m = re.fullmatch(r"\$\{(\w+)\}", obj)
        if m:
            if m.group(1) not in params:
                raise KeyError(f"undefined parameter ${{{m.group(1)}}}")
            return params[m.group(1)]

        def repl(mm: re.Match[str]) -> str:
            if mm.group(1) not in params:
                raise KeyError(f"undefined parameter ${{{mm.group(1)}}}")
            return str(params[mm.group(1)])

        return re.sub(r"\$\{(\w+)\}", repl, obj)
    if isinstance(obj, list):
        return [substitute(o, params) for o in obj]
    if isinstance(obj, dict):
        return {k: substitute(v, params) for k, v in obj.items()}
    return obj


def build_check(spec: dict[str, Any], params: dict[str, Any] | None = None) -> Check:
    spec = substitute(dict(spec), params or {})
    kind = spec.pop("type", None)
    if kind is None:
        raise ValueError(f"check is missing 'type': {spec}")
    if kind not in _REGISTRY:
        raise ValueError(f"unknown check type '{kind}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[kind](**spec)


def known_check_types() -> list[str]:
    return sorted(_REGISTRY)
