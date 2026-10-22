"""Turn a real agent's activity into a `Trajectory`.

Two ways in:

1. Wrap your tool functions with a `Recorder` and call `finish(answer)` at the end:

       rec = Recorder("refund-ok-A1001")
       lookup_order = rec.tool(lookup_order)       # or @rec.tool
       ... run your agent, which calls the wrapped tools ...
       traj = rec.finish(final_answer)

2. Convert an existing chat transcript in OpenAI message format (assistant messages with
   `tool_calls`, followed by `tool` messages) with `trajectory_from_openai_messages`.
"""

from __future__ import annotations

import functools
import json
import time
from collections.abc import Callable
from typing import Any

from agentgate.trajectory import ToolCall, Trajectory


class Recorder:
    def __init__(self, task_id: str, trial: int = 0):
        self._traj = Trajectory(task_id=task_id, trial=trial)

    def tool(self, fn: Callable[..., Any], name: str | None = None) -> Callable[..., Any]:
        label = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            call_args = dict(kwargs)
            if args:  # record positional args under their parameter names when we can
                import inspect

                params = list(inspect.signature(fn).parameters)
                call_args.update(dict(zip(params, args)))
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                self._traj.calls.append(ToolCall(
                    label, call_args, None, f"{type(exc).__name__}: {exc}",
                    latency_ms=(time.perf_counter() - start) * 1000,
                ))
                raise
            self._traj.calls.append(ToolCall(
                label, call_args, result, None,
                tokens=len(json.dumps(call_args, default=str)) // 4 + len(json.dumps(result, default=str)) // 4,
                latency_ms=(time.perf_counter() - start) * 1000,
            ))
            return result

        return wrapper

    def finish(self, answer: str, **meta: Any) -> Trajectory:
        self._traj.answer = answer
        self._traj.answer_tokens = int(len(answer.split()) * 1.3)
        self._traj.meta.update(meta)
        return self._traj


def trajectory_from_openai_messages(task_id: str, messages: list[dict[str, Any]], trial: int = 0) -> Trajectory:
    """Convert a chat transcript. The final assistant message with text is taken as the answer."""
    traj = Trajectory(task_id=task_id, trial=trial)
    pending: dict[str, ToolCall] = {}
    answer = ""
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                except json.JSONDecodeError:
                    args = {"_unparsed": raw}
                call = ToolCall(name=fn.get("name", "?"), args=args)
                traj.calls.append(call)
                pending[tc.get("id", "")] = call
            if msg.get("content"):
                answer = msg["content"]
        elif role == "tool":
            call = pending.get(msg.get("tool_call_id", ""))
            if call is not None:
                content = msg.get("content")
                try:
                    call.result = json.loads(content) if isinstance(content, str) else content
                except json.JSONDecodeError:
                    call.result = content
                if isinstance(call.result, dict) and call.result.get("error"):
                    call.error = str(call.result["error"])
                call.tokens = len(json.dumps(call.args, default=str)) // 4 + len(str(content)) // 4
    traj.answer = answer
    traj.answer_tokens = int(len(answer.split()) * 1.3)
    return traj
