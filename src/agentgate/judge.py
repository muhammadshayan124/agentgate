"""LLM-as-judge backends behind one tiny interface.

Some outputs cannot be checked with rules ("is this answer faithful to the tool results?"),
so a model is asked. That judge is itself an instrument and must be calibrated against
labels before anyone trusts a number it produces: see `calibrate.py`.

Backends:
- `HeuristicJudge`: lexical overlap, no model. A deliberately weak baseline to calibrate
  against, and a fallback when no model is available.
- `OllamaJudge`: a local model via the Ollama HTTP API (no key, no cost).
- `GeminiJudge`: Google's Gemini REST API (free tier works; needs GEMINI_API_KEY).

HTTP goes through an injectable `transport`, so the parsing and error handling are unit
tested without a network.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Example:
    id: str
    question: str
    evidence: str  # what the tools actually returned
    answer: str  # what the agent said
    label: bool  # ground truth: is the answer correct given the evidence?
    kind: str = ""
    source: str = "constructed"  # "constructed" (label known by construction) or "human"


@dataclass
class Verdict:
    correct: bool
    rationale: str = ""


class Judge(Protocol):
    name: str

    def judge(self, ex: Example) -> Verdict: ...


class JudgeError(RuntimeError):
    pass


PROMPT = """You are grading an AI support agent's answer against the facts its tools returned.

Customer question:
{question}

Facts returned by the agent's tools (the only source of truth):
{evidence}

Agent's answer:
{answer}

Is the agent's answer correct and consistent with the facts? An answer is incorrect if it
contradicts the facts, states something the facts do not support, or gets a number, status or
decision wrong. Ignore tone and length.

Reply with JSON only: {{"correct": true or false, "reason": "<one short sentence>"}}"""


def parse_verdict(text: str) -> Verdict:
    """Tolerates code fences and prose around the JSON; raises JudgeError if no verdict found."""
    cleaned = text.strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj.get("correct"), bool):
                return Verdict(obj["correct"], str(obj.get("reason", "")))
        except json.JSONDecodeError:
            pass
    low = cleaned.lower()
    if re.search(r'"?correct"?\s*[:=]\s*true', low):
        return Verdict(True, cleaned[:120])
    if re.search(r'"?correct"?\s*[:=]\s*false', low):
        return Verdict(False, cleaned[:120])
    raise JudgeError(f"could not parse a verdict from: {cleaned[:120]!r}")


class HeuristicJudge:
    """Marks an answer correct when most of its content words appear in the evidence.

    Cheap and wrong in instructive ways: it cannot see negation ("not eligible" vs "eligible").
    """

    name = "heuristic"

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t for t in re.findall(r"[a-z0-9$.]+", text.lower()) if len(t) >= 3}

    def judge(self, ex: Example) -> Verdict:
        ans, ev = self._tokens(ex.answer), self._tokens(ex.evidence + " " + ex.question)
        if not ans:
            return Verdict(False, "empty answer")
        overlap = len(ans & ev) / len(ans)
        return Verdict(overlap >= self.threshold, f"overlap={overlap:.2f}")


Transport = Callable[[str, dict[str, Any], dict[str, str]], dict[str, Any]]


def urllib_transport(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **headers}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - fixed https/http URLs we construct
        return json.loads(resp.read())


class OllamaJudge:
    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434", transport: Transport | None = None):
        self.name = f"ollama:{model}"
        self.model, self.host = model, host.rstrip("/")
        self._post = transport or urllib_transport

    def judge(self, ex: Example) -> Verdict:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [{"role": "user", "content": PROMPT.format(question=ex.question, evidence=ex.evidence, answer=ex.answer)}],
        }
        try:
            data = self._post(f"{self.host}/api/chat", payload, {})
            return parse_verdict(data["message"]["content"])
        except JudgeError:
            raise
        except Exception as exc:
            raise JudgeError(f"ollama request failed: {type(exc).__name__}: {exc}") from exc


class GeminiJudge:
    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None, transport: Transport | None = None):
        self.name = f"gemini:{model}"
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._post = transport or urllib_transport

    def judge(self, ex: Example) -> Verdict:
        if not self.api_key:
            raise JudgeError("GEMINI_API_KEY is not set")
        payload = {
            "contents": [{"parts": [{"text": PROMPT.format(question=ex.question, evidence=ex.evidence, answer=ex.answer)}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        try:
            data = self._post(url, payload, {"x-goog-api-key": self.api_key})
            return parse_verdict(data["candidates"][0]["content"]["parts"][0]["text"])
        except JudgeError:
            raise
        except Exception as exc:
            raise JudgeError(f"gemini request failed: {type(exc).__name__}") from exc  # never echo the key


def make_judge(kind: str, model: str | None = None) -> Judge:
    if kind == "heuristic":
        return HeuristicJudge()
    if kind == "ollama":
        return OllamaJudge(model or "llama3.2")
    if kind == "gemini":
        return GeminiJudge(model or "gemini-2.0-flash")
    raise ValueError("judge must be one of: heuristic, ollama, gemini")
