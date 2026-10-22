from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from agentgate import mde as mde_mod
from agentgate.calibrate import calibrate as run_calibration
from agentgate.calibrate import calibration_markdown, load_examples
from agentgate.checks import known_check_types
from agentgate.gate import GateConfig, compare
from agentgate.judge import make_judge
from agentgate.report import gate_markdown, power_markdown, run_summary
from agentgate.runner import RunResult, load_agent, run_suite, save_trajectories
from agentgate.suite import Suite

app = typer.Typer(help="Evaluation harness for tool-using agents: run suites, gate regressions, calibrate judges.")
console = Console(highlight=False)


@app.command()
def run(
    suite: Annotated[Path, typer.Argument(help="Suite YAML")],
    agent: Annotated[str, typer.Option(help="module:factory returning an object with .name and .run(task, seed)")],
    arg: Annotated[list[str] | None, typer.Option(help="key=value passed to the factory (repeatable)")] = None,
    trials: Annotated[int, typer.Option(help="Trials per task")] = 10,
    seed: int = 0,
    out: Annotated[Path, typer.Option(help="Where to write the run JSON")] = Path("run.json"),
    trajectories: Annotated[Path | None, typer.Option(help="Also write every trajectory (JSONL)")] = None,
) -> None:
    """Run an agent over a suite and save the result."""
    kwargs = dict(a.split("=", 1) for a in (arg or []))
    agent_obj = load_agent(agent, **kwargs)
    kept: list = [] if trajectories else None  # type: ignore[assignment]
    result = run_suite(Suite.load(suite), agent_obj, trials=trials, seed=seed, keep_trajectories=kept)
    result.save(out)
    if trajectories:
        save_trajectories(trajectories, kept)
    console.print(run_summary(result))
    console.print(f"saved {out}")


@app.command("compare")
def compare_runs(
    baseline: Path,
    candidate: Path,
    alpha: float = 0.05,
    min_effect: Annotated[float, typer.Option(help="Smallest absolute pass-rate drop worth failing for")] = 0.0,
    max_cost_increase: Annotated[float, typer.Option(help="Allowed relative rise in steps/tokens")] = 0.20,
    markdown: Annotated[Path | None, typer.Option(help="Write the report as Markdown (e.g. for a PR comment)")] = None,
) -> None:
    """Compare a candidate run to a baseline. Exits 1 if the gate fails."""
    cfg = GateConfig(alpha=alpha, min_effect=min_effect, max_cost_increase=max_cost_increase)
    rep = compare(RunResult.load(baseline), RunResult.load(candidate), cfg)
    md = gate_markdown(rep)
    console.print(md)
    if markdown:
        markdown.write_text(md, encoding="utf-8")
    raise typer.Exit(0 if rep.passed else 1)



@app.command()
def power(
    baseline: Path,
    shape: Annotated[str, typer.Option(help="uniform | concentrated")] = "uniform",
    sims: int = 300,
    alpha: float = 0.05,
) -> None:
    """How small a regression can this suite detect? Simulated from the baseline's own pass rates."""
    base = RunResult.load(baseline)
    drops = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
    pts = mde_mod.power_curve(base, drops, shape=shape, alpha=alpha, sims=sims)
    console.print(power_markdown(pts, shape, mde_mod.minimum_detectable_drop(pts)))


@app.command()
def calibrate(
    examples: Annotated[Path, typer.Argument(help="JSONL of labelled examples")],
    judge: Annotated[str, typer.Option(help="heuristic | ollama | gemini")] = "heuristic",
    model: Annotated[str | None, typer.Option(help="Model name for ollama/gemini")] = None,
    min_kappa: float = 0.6,
) -> None:
    """Measure a judge against labelled examples. Exits 1 if it is not usable."""
    report = run_calibration(make_judge(judge, model), load_examples(examples), min_kappa=min_kappa)
    console.print(calibration_markdown(report))
    raise typer.Exit(0 if report.usable else 1)


@app.command("checks")
def list_checks() -> None:
    """List the available check types."""
    for t in known_check_types():
        console.print(t)


@app.command("export-suite")
def export_suite(out: Path = Path("support_desk.yaml")) -> None:
    """Write the built-in simulated support-desk suite as YAML."""
    from agentgate.sim import build_suite

    out.write_text(build_suite().to_yaml(), encoding="utf-8")
    console.print(f"wrote {out}")


@app.command()
def demo(
    trials: int = 10,
    check: Annotated[bool, typer.Option(help="Exit 1 unless the gate behaves as expected")] = False,
) -> None:
    """Run the whole story on the simulated agent: an unchanged build passes, planted bugs fail."""
    from agentgate.demo import run_demo

    ok = run_demo(console, trials=trials)
    if check and not ok:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
