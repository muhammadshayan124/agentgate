import pytest

from agentgate.gate import GateConfig, compare
from agentgate.report import gate_markdown
from tests.helpers import make_run, uniform


def test_identical_runs_pass():
    r = make_run(uniform(24, 9))
    rep = compare(r, r)
    assert rep.passed and rep.quality.p_value == 1.0


def test_a_clear_regression_fails_with_a_reason():
    rep = compare(make_run(uniform(24, 9)), make_run(uniform(24, 5)))
    assert not rep.passed
    assert any("pass rate fell" in r for r in rep.reasons)
    assert rep.quality.failed and rep.quality.diff == pytest.approx(0.4)


def test_improvement_never_fails():
    assert compare(make_run(uniform(24, 5)), make_run(uniform(24, 9))).passed


def test_a_tiny_chance_dip_does_not_fail():
    base = uniform(24, 9)
    cand = dict(base)
    cand["t00"] = 8  # one task lost a single trial
    rep = compare(make_run(base), make_run(cand))
    assert rep.passed, rep.reasons
    assert rep.quality.diff > 0  # a naive "any drop" gate would have fired here


def test_min_effect_requires_a_big_enough_drop():
    base, cand = make_run(uniform(24, 9)), make_run(uniform(24, 8))
    assert not compare(base, cand, GateConfig()).passed
    assert compare(base, cand, GateConfig(min_effect=0.15)).passed


def test_cost_blowup_fails_even_if_quality_is_unchanged():
    rep = compare(make_run(uniform(24, 9), steps=4, tokens=300), make_run(uniform(24, 9), steps=8, tokens=300))
    assert not rep.passed and rep.steps.failed and not rep.tokens.failed
    assert any("steps rose" in r for r in rep.reasons)


def test_small_cost_increase_within_budget_passes():
    assert compare(make_run(uniform(24, 9), steps=10), make_run(uniform(24, 9), steps=11)).passed


def test_one_new_critical_violation_fails_regardless_of_statistics():
    base = make_run(uniform(24, 9))
    cand = make_run(uniform(24, 9), critical={"t03": 1})
    rep = compare(base, cand)
    assert not rep.passed and any("critical" in r for r in rep.reasons)
    assert rep.critical_base == 0 and rep.critical_cand == 1


def test_critical_failures_already_in_the_baseline_are_not_new():
    base = make_run(uniform(24, 9), critical={"t03": 1})
    assert compare(base, make_run(uniform(24, 9), critical={"t03": 1})).passed
    assert not compare(base, make_run(uniform(24, 9), critical={"t03": 2})).passed


def test_critical_allowance():
    cand = make_run(uniform(24, 9), critical={"t03": 1})
    assert compare(make_run(uniform(24, 9)), cand, GateConfig(critical_allowance=1)).passed


def test_runs_must_share_tasks():
    with pytest.raises(ValueError, match="share no tasks"):
        compare(make_run({"a": 5}), make_run({"b": 5}))


def test_only_common_tasks_are_compared():
    rep = compare(make_run({**uniform(23, 9), "only_base": 9}), make_run({**uniform(23, 9), "only_cand": 0}))
    assert rep.n_tasks == 23 and rep.passed


def test_report_names_the_worst_tasks_and_checks():
    base = uniform(24, 9)
    cand = dict(base)
    cand["t05"] = 0
    rep = compare(make_run(base), make_run(cand))
    assert rep.worst_tasks[0][0] == "t05"
    assert rep.worst_checks and rep.worst_checks[0][0] == "some_check"


def test_markdown_report_contents():
    rep = compare(make_run(uniform(24, 9)), make_run(uniform(24, 4), critical={"t01": 2}))
    md = gate_markdown(rep)
    assert md.startswith("### agentgate: FAIL")
    assert "pass rate" in md and "Most regressed tasks" in md and "critical-check failures" in md
    assert "### agentgate: PASS" in gate_markdown(compare(make_run(uniform(24, 9)), make_run(uniform(24, 9))))


def test_gate_is_deterministic_for_a_given_seed():
    a, b = make_run(uniform(24, 9)), make_run(uniform(24, 8))
    r1, r2 = compare(a, b, GateConfig(seed=7)), compare(a, b, GateConfig(seed=7))
    assert r1.quality.p_value == r2.quality.p_value and r1.quality.ci_lo == r2.quality.ci_lo


# -- the per-task rule ---------------------------------------------------------------------------

def test_one_completely_broken_task_is_caught_by_the_task_rule_but_invisible_to_the_mean():
    base = uniform(24, 9)
    cand = dict(base)
    cand["t07"] = 0
    with_rule = compare(make_run(base), make_run(cand))
    assert not with_rule.passed and with_rule.task_regressions[0][0] == "t07"
    assert any("regressed individually" in r for r in with_rule.reasons)
    # Without the rule the mean-based test sees p = 0.5 and waves it through: the hole this rule closes.
    assert compare(make_run(base), make_run(cand), GateConfig(task_rule=False)).passed


def test_task_rule_ignores_chance_wobble_across_many_tasks():
    base = uniform(24, 9)
    cand = {t: 8 if i % 2 else 9 for i, t in enumerate(base)}  # half the tasks lose one trial
    assert compare(make_run(base), make_run(cand)).task_regressions == []


def test_task_rule_needs_a_big_drop_not_just_a_significant_one():
    base = {**uniform(24, 27), "t00": 27}
    cand = {**base, "t00": 15}  # 90% -> 50%: significant with 30 trials, but only a 40-point drop
    rep = compare(make_run(base, trials=30), make_run(cand, trials=30))
    assert rep.task_regressions == []
    assert compare(make_run(base, trials=30), make_run(cand, trials=30), GateConfig(task_min_drop=0.2)).task_regressions


def test_task_rule_holm_correction_across_tasks():
    # 24 tasks each 10 -> 0: every one clears even the Holm-corrected threshold; all must be reported
    rep = compare(make_run(uniform(24, 10)), make_run(uniform(24, 0)))
    assert len(rep.task_regressions) == 24
    # 10 -> 4 is a real but modest drop per task; with only 10 trials it does NOT survive correction
    # across 24 tasks. That is the cost of controlling false alarms, and why more trials matter.
    assert compare(make_run(uniform(24, 10)), make_run(uniform(24, 4))).task_regressions == []
