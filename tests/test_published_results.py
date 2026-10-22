"""The README's headline claims, checked against the committed raw results.

If someone re-runs the experiment and the numbers move, these fail and the README must be
updated deliberately rather than drifting away from the data.
"""

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="module")
def summary():
    data = json.loads((RESULTS / "seeded_bugs.json").read_text(encoding="utf-8"))
    return data, {(e["bug"], e["rate"]): e for e in data["summary"]}


def test_experiment_size_matches_the_readme(summary):
    data, _ = summary
    assert data["reps"] == 200 and data["trials"] == 10


def test_false_alarm_rates(summary):
    _, s = summary
    aa = s[("none", 0.0)]
    assert aa["naive"]["rate"] > 0.30  # the naive gate is unusable
    assert aa["agentgate"]["rate"] <= 0.06 and aa["agentgate"]["hi"] < 0.10
    assert aa["agentgate"]["rate"] == aa["agentgate_core"]["rate"]  # the task rule added no false alarms


def test_the_mean_based_rules_miss_a_single_broken_task_and_the_task_rule_catches_it(summary):
    _, s = summary
    one = s[("broken_1_task", 1.0)]
    assert one["agentgate_core"]["rate"] < 0.20
    assert one["agentgate"]["rate"] >= 0.95
    assert one["by_task"]["rate"] >= 0.95


def test_safety_bugs_are_caught_by_the_critical_rule_even_when_the_mean_barely_moves(summary):
    _, s = summary
    for bug in ("refund_forbidden_orders", "missed_escalation", "skip_policy_check"):
        e = s[(bug, 0.35)]
        assert e["by_critical"]["rate"] >= 0.95, bug
    assert s[("missed_escalation", 0.35)]["mean_drop"] < 0.05


def test_trials_sensitivity_claim():
    t = json.loads((RESULTS / "trials_sensitivity.json").read_text(encoding="utf-8"))
    assert t["3"]["none"]["threshold"]["rate"] > 0.08  # a fixed threshold's false alarms grow as trials shrink
    for k in ("3", "5", "10", "20"):
        assert t[k]["none"]["agentgate"]["rate"] <= 0.06, k
