# agentgate

An evaluation harness for tool-using agents, built around one question: **when a change makes
your agent's score go down, is it a regression or just noise?**

Agents are stochastic. Run the *same* build twice and the pass rates differ. So "fail the PR if
the new score is lower" fires on a large share of identical builds, and "fail if it drops 5
points" only works for the one suite size it was tuned on. `agentgate` records what an agent
did (a *trajectory*), checks it with deterministic rules, and gates changes with statistics
whose false-alarm rate is controlled and *measured*.

> **Scope, stated up front.** Everything measured below uses a simulated agent (a deterministic
> stand-in with realistic flakiness and nine injectable bugs), because that is the only way to
> run the same agent twice for free, or to plant a bug of known size. The harness is
> agent-agnostic, but **no real LLM agent has been evaluated with it yet**, and no real LLM judge
> has been run (see [Limitations](#limitations)).

## What was measured

200 replications, 24 tasks, 10 trials per task. Each replication draws a fresh baseline, then
compares it to an unchanged build (A/A) and to the agent with a planted bug.

**Does the gate stay quiet when nothing changed?**

| gate | fires on an unchanged build | 95% CI |
|---|---|---|
| naive: fail on any drop in mean pass rate | **43%** | 36% to 50% |
| fail on a drop of 5+ points | 2% | 1% to 6% |
| agentgate (all four rules) | **3%** | 1.4% to 6.4% |

**Does it catch planted bugs?** (share of replications in which the gate fired; strength 0.35 =
the bug triggers on 35% of the trials it can affect. Full table: [`results/seeded_bugs.md`](results/seeded_bugs.md))

| planted bug | mean pass-rate drop | naive | 5-pt threshold | agentgate, mean-based rules only | agentgate, all four rules |
|---|---|---|---|---|---|
| skips the refund-policy check | 15.2% | 100% | 100% | 100% | 100% |
| refunds forbidden orders (safety) | 7.6% | 100% | 80% | 100% | 100% |
| refunds an angry customer instead of escalating (safety) | 4.3% | 92% | 40% | 100% | 100% |
| runaway loop (cost blow-up) | 27.7% | 100% | 100% | 100% | 100% |
| wrong refund amount | 7.8% | 99% | 84% | 59% | 68% |
| states the wrong order status | 8.3% | 100% | 86% | 55% | 56% |
| bad knowledge-base query (3 of 24 tasks) | 3.5% | 89% | 32% | 16% | 19% |
| **one task completely broken** | 4.0% | 92% | 38% | **8%** | **100%** |
| three tasks completely broken | 11.4% | 100% | 100% | 36% | 100% |

What this shows, including the parts that don't flatter the tool:

- **The naive gate is useless**: it catches everything and also fails 43% of builds that changed nothing.
- **The safety rule is the strongest signal.** "Critical" checks (must-never-happen rules) catch
  both safety bugs 100% of the time even when the average pass rate barely moves (4% drop).
- **A mean-based test is nearly blind to one broken task.** One task going 100% to 0% gave a
  detection rate of 8%. That finding is why the fourth rule exists (a per-task exact test with
  Holm correction), which closed it to 100% with no increase in false alarms (3% either way).
- **The fixed 5-point threshold is more sensitive than agentgate on some diffuse bugs**
  (`wrong_refund_amount` 84% vs 68%, `states the wrong status` 86% vs 56%). That is not free:
  a threshold is only right for the suite size it was tuned on. Changing trials per task:

| trials per task | 5-pt threshold fires on unchanged build | agentgate fires on unchanged build | 5-pt threshold detects wrong-amount bug | agentgate detects it |
|---|---|---|---|---|
| 3 | 14% | 2% | 68% | 22% |
| 5 | 9% | 3% | 72% | 34% |
| 10 | 2% | 3% | 83% | 60% |
| 20 | 0% | 3% | 89% | 85% |

agentgate's false-alarm rate does not move; the threshold's does. The price of that control is
power at low trial counts (22% at 3 trials). Use `agentgate power` to see what *your* suite can
detect before trusting a green build.

## Quick start

With a virtual environment:

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                                             # 100+ tests
agentgate demo --check                             # the gate must pass an unchanged build and fail planted bugs
```

With Docker:

```bash
docker compose run --rm test        # test suite in the image
docker compose run --rm demo        # the same self-check
docker compose run --rm gate-example   # baseline vs a buggy candidate, writes out/report.md
docker compose --profile experiment run --rm experiment   # reproduce the tables above (~10 min)
```

## How it works

```
agent --run--> trajectory (tool calls + answer) --checks--> pass/fail per trial
                                                                  |
        baseline run  --\                                        per-task pass rates
                         >-- compare --> 4 rules --> pass / fail + the reasons
        candidate run --/
```

**Checks** are declared in YAML and evaluated on the trajectory: `tool_called` (with argument
matching), `tool_not_called`, `order`, `max_steps`, `max_tokens`, `no_repeat_calls`,
`no_tool_errors`, `answer_contains`, `answer_not_contains`, `answer_regex`. Any check can be
`critical: true` for a rule that must never be broken. See [`examples/support_desk.yaml`](examples/support_desk.yaml).

**The gate** (`agentgate compare baseline.json candidate.json`) applies four independent rules
and reports which one failed:

1. **Quality**: the paired per-task drop in pass rate is significant under a one-sided sign-flip
   randomisation test (exact in size, no normality assumption). The unit is the *task*, not the
   trial, because trials of one task are correlated.
2. **Cost**: steps and tokens may not rise by more than 20% (and significantly), so an agent cannot
   "improve" by becoming much more expensive.
3. **Critical**: a must-never check may not fail more often than in the baseline. No statistics:
   one new violation is enough.
4. **Task**: any single task with a large drop that survives an exact test with Holm correction
   across all tasks.

`agentgate power baseline.json` simulates how big a regression the suite can actually detect.

## Using it on your own agent

Wrap your tools, or convert an OpenAI-format transcript:

```python
from agentgate.recorder import Recorder, trajectory_from_openai_messages

rec = Recorder("refund-ok-A1001", trial=0)
lookup_order = rec.tool(lookup_order)          # your function, now recorded
...                                            # run your agent
traj = rec.finish(final_answer)

traj = trajectory_from_openai_messages("refund-ok-A1001", messages)
```

Or expose a factory returning an object with `.name` and `.run(task, seed) -> Trajectory` and run
`agentgate run suite.yaml --agent my_package:make_agent --trials 10 --out candidate.json`.
Gate a pull request with the included composite action (`action.yml`), which writes the report to
the job summary and fails the check.

## Judge calibration

Some outputs cannot be checked with rules, so a model is asked. That judge is an instrument and
needs calibrating before its scores gate anything. `agentgate calibrate examples.jsonl --judge
{heuristic,ollama,gemini}` reports accuracy and Cohen's kappa with bootstrap intervals, false
accept / false reject rates (what matters for a gate), and a verbosity probe (does padding a wrong
answer with confident filler make it look right?). A judge is "usable" only if the *lower* bound
of kappa clears the bar **and** at least 50 examples were scored; a lucky small sample cannot certify it.

What has actually been run: the deliberately weak lexical baseline scores kappa 0.43 (95% CI
0.26 to 0.59) on the bundled set and is correctly rejected. The Ollama and Gemini backends are
unit-tested against a fake HTTP transport (request shape, parsing, error handling, the API key
never appears in a URL or error message) but **have not been run against a real model**.

## Limitations

- **Simulated subject.** Every number above comes from the simulated support-desk agent. It
  models flakiness and task difficulty but not language or reasoning. The false-alarm control
  rests on assumptions (below) that real agents may break.
- **Assumptions of the quality test**: per-task differences are symmetric around zero for an
  unchanged build, and tasks are independent. Shared state, caching, provider rate limits or
  model-version drift across a run can violate this. Inference is about *this suite*, not about
  unseen tasks.
- **The judge dataset is constructed, not human-labelled.** Labels are exact by construction
  (wrong answers are templated corruptions of known facts), so it is easier than real model
  output; each example carries `source: constructed` and reports say so. A judge that fails it is
  unusable; one that passes has cleared a low bar.
- **Power is limited at low trial counts.** 10 trials per task cannot flag a modest per-task drop
  after multiple-comparison correction (10/10 to 4/10 on one task is *not* flagged). Partial
  regressions confined to a few tasks (`bad_kb_query`) are often missed. Add trials, or make the
  behaviour a critical check.
- **One baseline sample.** The comparison uses a single baseline run, so its noise is part of the
  test (as it should be), but a better baseline (more trials, or pooled across commits) increases power.

## Layout

```
src/agentgate/   trajectory, checks, suite, runner, stats, gate, mde, judge, calibrate, recorder, report, cli
src/agentgate/sim/          simulated environment, agent (9 bugs), task suite, judge examples
src/agentgate/experiments/  the measurements above
examples/ data/ results/    suite YAML, judge examples, raw results (JSON + Markdown)
tests/                      100+ tests, including the statistical properties
```

## License

MIT
