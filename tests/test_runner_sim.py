from pathlib import Path

import pytest

from agentgate.runner import RunResult, load_agent, run_suite, trial_seed
from agentgate.sim import BUGS, build_suite, make_agent
from agentgate.sim.env import ORDERS, Env
from agentgate.suite import Suite, Task

ROOT = Path(__file__).resolve().parents[1]
PINNED_SEED = 9996024662810770434  # SHA-256 based, so identical on every OS and Python version


def test_trial_seed_is_stable_and_distinct():
    assert trial_seed(1, "task", 0) == trial_seed(1, "task", 0)
    seeds = {trial_seed(1, t, i) for t in ("a", "b") for i in range(5)}
    assert len(seeds) == 10
    assert trial_seed(1, "a", 0) != trial_seed(2, "a", 0)
    assert trial_seed(0, "refund-ok-A1001", 0) == PINNED_SEED


def test_example_yaml_matches_the_code_that_generates_it():
    assert (ROOT / "examples" / "support_desk.yaml").read_text(encoding="utf-8") == build_suite().to_yaml()


def test_suite_loads_validates_and_rejects_duplicates(tmp_path):
    s = Suite.load(ROOT / "examples" / "support_desk.yaml")
    assert len(s.tasks) == 24 and {"refund", "safety", "kb", "status", "escalation"} <= {t for x in s.tasks for t in x.tags}
    with pytest.raises(ValueError, match="duplicate"):
        Suite("x", [Task("a", ""), Task("a", "")])
    bad = Task("b", "", check_specs=[{"type": "nope"}])
    with pytest.raises(ValueError, match="unknown check type"):
        Suite("x", [bad])


def test_runs_are_reproducible_and_seed_sensitive():
    s = build_suite()
    a = run_suite(s, make_agent(), trials=5, seed=3)
    b = run_suite(s, make_agent(), trials=5, seed=3)
    c = run_suite(s, make_agent(), trials=5, seed=4)
    assert a.results == b.results
    assert a.results != c.results


def test_baseline_is_good_but_not_perfect_and_never_violates_safety():
    r = run_suite(build_suite(), make_agent(), trials=30, seed=11)
    assert 0.80 < r.pass_rate < 0.97  # flaky on purpose
    assert r.critical_failures == 0
    rates = list(r.task_rates().values())
    assert max(rates) > min(rates)  # tasks differ in difficulty


def test_run_result_round_trips_through_json(tmp_path):
    r = run_suite(build_suite(), make_agent("hallucinate_status", 0.5), trials=3, seed=1)
    path = tmp_path / "run.json"
    r.save(path)
    again = RunResult.load(path)
    assert again.results == r.results and again.agent == r.agent and again.pass_rate == r.pass_rate


def test_trajectories_can_be_kept_and_match_results():
    kept = []
    r = run_suite(build_suite(), make_agent(), trials=2, seed=1, keep_trajectories=kept)
    assert len(kept) == len(r.results) == 48
    assert kept[0].steps == r.results[0].steps and kept[0].tokens == r.results[0].tokens


def test_load_agent_coerces_cli_string_values():
    a = load_agent("agentgate.sim:make_agent", bug="skip_policy_check", rate="0.4", noise="2")
    assert a.bug == "skip_policy_check" and a.rate == 0.4 and a.noise == 2.0
    with pytest.raises(ValueError, match="module:factory"):
        load_agent("no_colon")


def test_unknown_bug_is_rejected():
    with pytest.raises(ValueError, match="unknown bug"):
        make_agent("telepathy", 0.5)


@pytest.mark.parametrize("bug", sorted(BUGS))
def test_every_planted_bug_makes_the_agent_measurably_worse(bug):
    s = build_suite()
    base = run_suite(s, make_agent(), trials=20, seed=5)
    bad = run_suite(s, make_agent(bug, 0.9), trials=20, seed=5)
    assert bad.pass_rate < base.pass_rate - 0.01, bug


def test_safety_bugs_trip_critical_checks_and_benign_ones_do_not():
    s = build_suite()
    for bug in ("skip_policy_check", "refund_forbidden_orders", "missed_escalation"):
        assert run_suite(s, make_agent(bug, 0.9), trials=10, seed=2).critical_failures > 0, bug
    for bug in ("wrong_refund_amount", "hallucinate_status", "bad_kb_query", "runaway_loop"):
        assert run_suite(s, make_agent(bug, 0.9), trials=10, seed=2).critical_failures == 0, bug


def test_a_retried_write_is_a_double_refund_but_a_retried_read_is_harmless():
    s = build_suite()
    # The repeated call is the *last* one. Normally that is the confirmation email, so a retry is
    # merely redundant; it becomes a double refund only when the email step was skipped.
    quiet = run_suite(s, make_agent("redundant_calls", 1.0, noise=0.0), trials=5, seed=1)
    assert quiet.critical_failures == 0
    slippy = run_suite(s, make_agent("redundant_calls", 1.0, noise=10.0), trials=10, seed=1)
    assert slippy.critical_failures > 0
    # The baseline's own benign slip only ever re-reads, so it can never double-refund.
    assert run_suite(s, make_agent(noise=3.0), trials=20, seed=1).critical_failures == 0


def test_env_tools_really_execute():
    env = Env()
    out, err = env.call("lookup_order", {"order_id": "A1001"})
    assert err is None and out["total"] == 59.90
    assert env.call("check_refund_policy", {"order_id": "B2001"})[0]["eligible"] is False
    assert env.call("check_refund_policy", {"order_id": "B2002"})[0]["reason"].startswith("final-sale")
    assert env.call("check_refund_policy", {"order_id": "B2004"})[0]["reason"].startswith("the order has not been delivered")
    assert env.call("lookup_order", {"order_id": "NOPE"})[1].startswith("ToolError")
    assert env.call("issue_refund", {"order_id": "A1001"})[1].startswith("TypeError")  # missing arg is an error, not a crash
    assert env.call("search_kb", {"query": "How long does SHIPPING take"})[0]["answer"].startswith("Standard shipping")
    assert env.call("search_kb", {"query": "weather"})[0]["answer"] is None
    assert env.call("escalate", {"reason": "x"})[0]["ticket"] == "T-1"
    assert all(o.customer_email.endswith("@example.test") for o in ORDERS.values())
