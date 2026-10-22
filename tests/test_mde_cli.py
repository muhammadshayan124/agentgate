import json

import pytest
from typer.testing import CliRunner

from agentgate import mde
from agentgate.cli import app
from agentgate.runner import RunResult
from tests.helpers import make_run, uniform

runner = CliRunner()


# -- power / minimum detectable effect ------------------------------------------------------

def test_power_rises_with_effect_size_and_starts_near_alpha():
    base = make_run(uniform(24, 9))
    pts = mde.power_curve(base, [0.0, 0.05, 0.15, 0.30], sims=120, n_perm=500, seed=1)
    powers = [p.power for p in pts]
    assert powers[0] < 0.15  # no real drop -> rarely fires
    assert powers == sorted(powers) or powers[-1] > powers[1]
    assert powers[-1] > 0.95


def test_more_trials_means_more_power_for_the_same_drop():
    few = mde.power_curve(make_run(uniform(24, 9), trials=3), [0.1], sims=150, n_perm=500, seed=2)[0].power
    many = mde.power_curve(make_run(uniform(24, 9), trials=30), [0.1], sims=150, n_perm=500, seed=2)[0].power
    assert many > few


def test_concentrated_shape_is_supported_and_validated():
    base = make_run(uniform(24, 9))
    conc = mde.power_curve(base, [0.1, 0.2], shape="concentrated", sims=150, n_perm=300, seed=4)
    uni = mde.power_curve(base, [0.1], shape="uniform", sims=150, n_perm=300, seed=4)
    assert conc[1].power > 0.9  # a big enough concentrated drop is eventually seen
    assert conc[0].power < uni[0].power  # but at equal mean drop it is harder than a diffuse one
    with pytest.raises(ValueError):
        mde.power_curve(base, [0.1], shape="weird")


def test_minimum_detectable_drop():
    pts = [mde.PowerPoint(0.02, 0.1), mde.PowerPoint(0.05, 0.85), mde.PowerPoint(0.1, 1.0)]
    assert mde.minimum_detectable_drop(pts) == 0.05
    assert mde.minimum_detectable_drop(pts[:1]) is None


# -- CLI --------------------------------------------------------------------------------------

def test_demo_check_passes():
    result = runner.invoke(app, ["demo", "--check"])
    assert result.exit_code == 0, result.output
    assert "as expected" in result.output


def test_checks_and_export_suite(tmp_path):
    assert "tool_called" in runner.invoke(app, ["checks"]).output
    out = tmp_path / "s.yaml"
    assert runner.invoke(app, ["export-suite", "--out", str(out)]).exit_code == 0
    assert out.read_text(encoding="utf-8").startswith("name: support-desk")


def test_run_then_compare_end_to_end(tmp_path):
    suite = "examples/support_desk.yaml"
    base, good, bad = (tmp_path / f"{n}.json" for n in ("base", "good", "bad"))
    traj = tmp_path / "t.jsonl"
    common = ["run", suite, "--agent", "agentgate.sim:make_agent", "--trials", "8"]  # fixed seeds: a single A/A pair fires about 3% of the time (see README)
    r = runner.invoke(app, [*common, "--seed", "1", "--out", str(base), "--trajectories", str(traj)])
    assert r.exit_code == 0, r.output
    assert len(traj.read_text(encoding="utf-8").splitlines()) == 24 * 8
    assert json.loads(traj.read_text(encoding="utf-8").splitlines()[0])["calls"]
    assert runner.invoke(app, [*common, "--seed", "3", "--out", str(good)]).exit_code == 0
    assert runner.invoke(app, [*common, "--seed", "2", "--arg", "bug=runaway_loop", "--arg", "rate=0.6", "--out", str(bad)]).exit_code == 0

    ok = runner.invoke(app, ["compare", str(base), str(good)])
    assert ok.exit_code == 0 and "agentgate: PASS" in ok.output

    md = tmp_path / "report.md"
    fail = runner.invoke(app, ["compare", str(base), str(bad), "--markdown", str(md)])
    assert fail.exit_code == 1 and "agentgate: FAIL" in fail.output
    assert md.read_text(encoding="utf-8").startswith("### agentgate: FAIL")
    assert RunResult.load(bad).agent == "sim+runaway_loop@0.6"


def test_power_command(tmp_path):
    base = tmp_path / "base.json"
    make_run(uniform(24, 9)).save(base)
    out = runner.invoke(app, ["power", str(base), "--sims", "40"])
    assert out.exit_code == 0 and "mean pass-rate drop" in out.output


def test_calibrate_command_exit_codes():
    out = runner.invoke(app, ["calibrate", "data/judge_examples.jsonl", "--judge", "heuristic"])
    assert out.exit_code == 1 and "NOT usable" in out.output  # the lexical baseline must be rejected
