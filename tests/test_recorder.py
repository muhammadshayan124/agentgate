import json

import pytest

from agentgate.checks import build_check
from agentgate.recorder import Recorder, trajectory_from_openai_messages


def test_recorder_captures_calls_args_results_and_errors():
    rec = Recorder("t1", trial=2)

    def lookup(order_id, verbose=False):
        return {"id": order_id}

    def boom(x):
        raise KeyError(x)

    lookup, boom = rec.tool(lookup), rec.tool(boom, name="explode")
    assert lookup("A1", verbose=True) == {"id": "A1"}  # wrapped function still behaves
    with pytest.raises(KeyError):
        boom("z")
    traj = rec.finish("all done here", model="m1")

    assert traj.task_id == "t1" and traj.trial == 2 and traj.meta == {"model": "m1"}
    assert [c.name for c in traj.calls] == ["lookup", "explode"]
    assert traj.calls[0].args == {"verbose": True, "order_id": "A1"} and traj.calls[0].result == {"id": "A1"}
    assert traj.calls[1].error.startswith("KeyError") and traj.calls[0].latency_ms >= 0
    assert traj.answer == "all done here" and traj.tokens > 0


def test_recorded_trajectory_works_with_checks():
    rec = Recorder("t")
    pay = rec.tool(lambda amount: "ok", name="pay")
    pay(amount=5)
    pay(amount=5)
    traj = rec.finish("paid")
    assert not build_check({"type": "no_repeat_calls"}).evaluate(traj).passed
    assert not build_check({"type": "tool_called", "tool": "pay", "max": 1}).evaluate(traj).passed


MESSAGES = [
    {"role": "user", "content": "Refund order A1001"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "lookup_order", "arguments": json.dumps({"order_id": "A1001"})}}]},
    {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"total": 59.9})},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c2", "type": "function", "function": {"name": "issue_refund", "arguments": "{bad json"}}]},
    {"role": "tool", "tool_call_id": "c2", "content": json.dumps({"error": "gateway timeout"})},
    {"role": "assistant", "content": "I could not complete the refund."},
]


def test_openai_transcript_conversion():
    traj = trajectory_from_openai_messages("t", MESSAGES, trial=1)
    assert [c.name for c in traj.calls] == ["lookup_order", "issue_refund"]
    assert traj.calls[0].args == {"order_id": "A1001"} and traj.calls[0].result == {"total": 59.9}
    assert traj.calls[1].args == {"_unparsed": "{bad json"}  # bad arguments are kept, not dropped or crashed on
    assert traj.calls[1].error == "gateway timeout"
    assert traj.answer == "I could not complete the refund." and traj.trial == 1


def test_conversion_tolerates_orphan_tool_messages_and_no_answer():
    traj = trajectory_from_openai_messages("t", [{"role": "tool", "tool_call_id": "zz", "content": "x"}])
    assert traj.calls == [] and traj.answer == ""
