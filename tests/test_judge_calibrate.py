import json
from pathlib import Path

import pytest

from agentgate.calibrate import FLUFF, calibrate, calibration_markdown, load_examples
from agentgate.judge import (
    Example,
    GeminiJudge,
    HeuristicJudge,
    JudgeError,
    OllamaJudge,
    Verdict,
    make_judge,
    parse_verdict,
)
from agentgate.sim.judge_data import build_examples

ROOT = Path(__file__).resolve().parents[1]


def ex(label=True, answer="Order A1 is shipped.", **kw):
    return Example(id="x", question="Where?", evidence="status: shipped", answer=answer, label=label, **kw)


# -- parsing -------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"correct": true, "reason": "matches"}', True),
        ('```json\n{"correct": false, "reason": "wrong status"}\n```', False),
        ('Sure! {"correct": true} hope that helps', True),
        ('correct: false', False),
    ],
)
def test_parse_verdict_variants(text, expected):
    assert parse_verdict(text).correct is expected


@pytest.mark.parametrize("text", ["", "I think it is fine", '{"correct": "maybe"}', "{not json"])
def test_parse_verdict_rejects_garbage(text):
    with pytest.raises(JudgeError):
        parse_verdict(text)


# -- backends (fake transport, no network) ---------------------------------------------------

def test_ollama_judge_request_and_response():
    seen = {}

    def transport(url, payload, headers):
        seen.update(url=url, payload=payload)
        return {"message": {"content": '{"correct": true, "reason": "ok"}'}}

    v = OllamaJudge("m", "http://h:1/", transport).judge(ex())
    assert v.correct and seen["url"] == "http://h:1/api/chat"
    assert seen["payload"]["options"]["temperature"] == 0 and seen["payload"]["stream"] is False
    assert "status: shipped" in seen["payload"]["messages"][0]["content"]


def test_ollama_connection_failure_becomes_judge_error():
    def transport(*a):
        raise ConnectionRefusedError("down")

    with pytest.raises(JudgeError, match="ollama request failed"):
        OllamaJudge(transport=transport).judge(ex())


def test_gemini_requires_a_key_and_never_leaks_it(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(JudgeError, match="not set"):
        GeminiJudge(transport=lambda *a: {}).judge(ex())

    secret = "AIza-test-secret-value"

    def transport(url, payload, headers):
        assert secret not in url and headers["x-goog-api-key"] == secret  # key travels in a header, not the URL
        raise RuntimeError(f"boom {secret}")

    with pytest.raises(JudgeError) as e:
        GeminiJudge(api_key=secret, transport=transport).judge(ex())
    assert secret not in str(e.value)


def test_gemini_response_parsing():
    payload = {"candidates": [{"content": {"parts": [{"text": '{"correct": false, "reason": "x"}'}]}}]}
    assert GeminiJudge(api_key="k", transport=lambda *a: payload).judge(ex()).correct is False


def test_make_judge():
    assert make_judge("heuristic").name == "heuristic"
    assert make_judge("ollama", "llama3").name == "ollama:llama3"
    with pytest.raises(ValueError):
        make_judge("magic")


# -- calibration --------------------------------------------------------------------------

class Oracle:
    name = "oracle"

    def judge(self, e):
        return Verdict(e.label)


class Flipper:
    name = "flipper"

    def judge(self, e):
        return Verdict(not e.label)


class AlwaysYes:
    name = "yes-man"

    def judge(self, e):
        return Verdict(True)


class Flaky:
    name = "flaky"

    def __init__(self):
        self.n = 0

    def judge(self, e):
        self.n += 1
        if self.n % 5 == 0:
            raise JudgeError("timeout")
        return Verdict(e.label)


DATA = build_examples()


def test_oracle_is_usable_and_flipper_is_not():
    good = calibrate(Oracle(), DATA)
    assert good.usable and good.kappa == 1.0 and good.accuracy == 1.0
    assert good.false_accept_rate == 0 and good.false_reject_rate == 0
    bad = calibrate(Flipper(), DATA)
    assert not bad.usable and bad.kappa < -0.8  # classes are imbalanced, so not exactly -1


def test_always_yes_has_zero_kappa_despite_decent_accuracy():
    r = calibrate(AlwaysYes(), DATA)
    assert abs(r.kappa) < 1e-9 and not r.usable
    assert r.false_accept_rate == 1.0 and r.false_reject_rate == 0.0
    assert r.accuracy > 0.3  # accuracy alone would flatter it; kappa does not


def test_abstentions_are_counted_not_hidden():
    r = calibrate(Flaky(), DATA, probe_verbosity=False)
    assert r.abstained > 0 and r.n + r.abstained == len(DATA)


def test_too_few_verdicts_is_an_error():
    with pytest.raises(ValueError, match="at least 10"):
        calibrate(Oracle(), DATA[:5])


def test_a_lucky_small_sample_cannot_certify_a_judge():
    r = calibrate(Oracle(), DATA[:10])
    assert r.kappa == 1.0 and not r.usable  # perfect on 10 examples is still not enough evidence
    assert calibrate(Oracle(), DATA[:10], min_n=10).usable
    assert calibrate(Oracle(), DATA, min_kappa=1.0).usable is True


def test_verbosity_probe_detects_a_judge_fooled_by_confident_filler():
    class Gullible:
        name = "gullible"

        def judge(self, e):
            return Verdict(e.label or "completely confident" in e.answer)

    r = calibrate(Gullible(), DATA)
    assert r.verbosity_delta == pytest.approx(1.0 - 0.0)
    assert "completely confident" in FLUFF
    assert calibrate(Oracle(), DATA).verbosity_delta == 0.0


def test_markdown_flags_non_human_labels_and_usability():
    md = calibration_markdown(calibrate(Oracle(), DATA))
    assert "USABLE" in md and "not (all) human-made" in md
    assert "NOT usable" in calibration_markdown(calibrate(Flipper(), DATA))
    human = [Example(**{**e.__dict__, "source": "human"}) for e in DATA]
    assert "not (all) human-made" not in calibration_markdown(calibrate(Oracle(), human))


# -- the dataset and the weak baseline judge ---------------------------------------------------

def test_dataset_is_in_sync_balanced_enough_and_has_unique_ids():
    path = ROOT / "data" / "judge_examples.jsonl"
    on_disk = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert on_disk == [e.__dict__ for e in DATA]
    assert len({e.id for e in DATA}) == len(DATA) >= 100
    positives = sum(e.label for e in DATA)
    assert 0.3 < positives / len(DATA) < 0.7
    assert {e.source for e in DATA} == {"constructed"}
    assert len(load_examples(path)) == len(DATA)


def test_the_lexical_baseline_is_rejected_as_a_gate_input():
    r = calibrate(HeuristicJudge(), DATA)
    assert not r.usable and 0.2 < r.kappa < 0.6
    assert HeuristicJudge().judge(ex(answer="")).correct is False
