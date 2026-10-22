import pytest

from agentgate.checks import args_match, build_check, known_check_types, substitute
from agentgate.trajectory import ToolCall, Trajectory


def traj(*calls, answer=""):
    return Trajectory("t", calls=[ToolCall(n, a) for n, a in calls], answer=answer, answer_tokens=len(answer.split()))


def run(spec, t, params=None):
    return build_check(spec, params).evaluate(t)


def test_tool_called_matches_args_as_a_subset():
    t = traj(("lookup", {"id": "A1", "verbose": True}))
    assert run({"type": "tool_called", "tool": "lookup", "args": {"id": "A1"}}, t).passed
    assert not run({"type": "tool_called", "tool": "lookup", "args": {"id": "ZZ"}}, t).passed


def test_tool_called_reports_what_was_actually_passed():
    t = traj(("lookup", {"id": "WRONG"}))
    r = run({"type": "tool_called", "tool": "lookup", "args": {"id": "A1"}}, t)
    assert not r.passed and "WRONG" in r.reason


def test_tool_called_min_and_max():
    t = traj(("pay", {}), ("pay", {}))
    assert not run({"type": "tool_called", "tool": "pay", "max": 1}, t).passed
    assert run({"type": "tool_called", "tool": "pay", "min": 0, "max": 2}, t).passed
    assert run({"type": "tool_called", "tool": "pay", "min": 0, "max": 1}, traj()).passed


def test_numbers_match_with_tolerance_but_strings_do_not_coerce():
    assert args_match({"amount": 59.9}, {"amount": 59.900000001})
    assert not args_match({"amount": 59.9}, {"amount": 65.89})
    assert not args_match({"id": "1"}, {"id": 1})


def test_tool_not_called():
    assert run({"type": "tool_not_called", "tool": "refund"}, traj(("lookup", {}))).passed
    assert not run({"type": "tool_not_called", "tool": "refund"}, traj(("refund", {}))).passed


def test_order_is_a_subsequence_not_exact_sequence():
    t = traj(("a", {}), ("x", {}), ("b", {}), ("c", {}))
    assert run({"type": "order", "tools": ["a", "b", "c"]}, t).passed
    bad = run({"type": "order", "tools": ["a", "c", "b"]}, t)
    assert not bad.passed and "first missing: b" in bad.reason


def test_order_with_missing_tool_fails():
    assert not run({"type": "order", "tools": ["a", "zzz"]}, traj(("a", {}))).passed


def test_budgets():
    t = traj(("a", {}), ("b", {}), answer="x " * 10)
    assert run({"type": "max_steps", "n": 2}, t).passed
    assert not run({"type": "max_steps", "n": 1}, t).passed
    assert not run({"type": "max_tokens", "n": 5}, t).passed


def test_no_repeat_calls_ignores_arg_order_and_distinguishes_args():
    assert not run({"type": "no_repeat_calls"}, traj(("a", {"x": 1, "y": 2}), ("a", {"y": 2, "x": 1}))).passed
    assert run({"type": "no_repeat_calls"}, traj(("a", {"x": 1}), ("a", {"x": 2}))).passed


def test_answer_checks_are_case_insensitive():
    t = traj(answer="Your Refund of $59.90 is on its way")
    assert run({"type": "answer_contains", "values": ["refund", "$59.90"]}, t).passed
    assert not run({"type": "answer_contains", "values": ["refund", "cancelled"]}, t).passed
    assert run({"type": "answer_contains", "values": ["cancelled", "refund"], "mode": "any"}, t).passed
    assert not run({"type": "answer_not_contains", "values": ["REFUND"]}, t).passed
    assert run({"type": "answer_regex", "pattern": r"\$\d+\.\d{2}"}, t).passed


def test_no_tool_errors():
    t = Trajectory("t", calls=[ToolCall("a", error="boom")])
    assert not build_check({"type": "no_tool_errors"}).evaluate(t).passed


def test_critical_flag_and_custom_name_propagate():
    r = run({"type": "tool_not_called", "tool": "refund", "critical": True, "name": "never_refund"}, traj(("refund", {})))
    assert r.critical and r.name == "never_refund" and not r.passed


def test_a_broken_check_fails_loudly_instead_of_passing():
    class Boom(build_check({"type": "max_steps", "n": 1}).__class__):
        def _evaluate(self, traj):
            raise RuntimeError("bug in check")

    r = Boom(n=1).evaluate(traj())
    assert not r.passed and "RuntimeError" in r.reason


def test_substitution_preserves_types_for_whole_placeholders():
    out = substitute({"amount": "${amount}", "msg": "pay ${amount} for ${oid}", "xs": ["${oid}"]}, {"amount": 59.9, "oid": "A1"})
    assert out == {"amount": 59.9, "msg": "pay 59.9 for A1", "xs": ["A1"]}


def test_undefined_parameter_is_an_error():
    with pytest.raises(KeyError):
        substitute("${nope}", {})


def test_unknown_and_missing_type_are_errors():
    with pytest.raises(ValueError, match="unknown check type"):
        build_check({"type": "telepathy"})
    with pytest.raises(ValueError, match="missing 'type'"):
        build_check({"tool": "x"})


def test_all_registered_types_are_listed():
    assert {"tool_called", "order", "max_steps", "answer_contains", "no_repeat_calls"} <= set(known_check_types())
