"""Shared test helpers. Run from the repo root with `python -m pytest`."""

import json
import os
from pathlib import Path

from promptevo.datasets.base import RepairResult
from promptevo.scoring.evaluation import Evaluation
from promptevo.scoring.individual import Individual

GOLDEN = Path(__file__).parent / "golden"


def check_golden(name, data):
    """compares data with tests/golden/<name>.json. UPDATE_GOLDEN=1 writes
    it instead, only do that when a change in output is on purpose."""
    path = GOLDEN / f"{name}.json"
    text = json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True)
    if os.environ.get("UPDATE_GOLDEN"):
        path.parent.mkdir(exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        return
    assert path.exists(), "no golden file, run with UPDATE_GOLDEN=1"
    assert json.loads(text) == json.loads(path.read_text(encoding="utf-8"))


class RecordingModel:
    """stands in for a chat model. remembers every message and replies
    with a numbered canned answer, or one from replies if given."""

    def __init__(self, replies=None):
        self.calls = 0
        self.sent = []
        self._replies = list(replies or [])

    def chat(self, user, system=None, seed=None):
        self.calls += 1
        self.sent.append({"system": system, "user": user, "seed": seed})
        if self._replies:
            return self._replies.pop(0)
        return f"New prompt: reply {self.calls}"

    def chat_many(self, users, system=None):
        return [self.chat(user, system) for user in users]


def attempt(item, score, output="def f():\n    return 1"):
    """a scored item with an attempt, for mutations that look at them."""
    return RepairResult(
        tests={f"{item}-a": score > 0, f"{item}-b": score == 1.0},
        inputs=f"fix {item}",
        output=output,
        to_fix=frozenset({f"{item}-a", f"{item}-b"}),
    )


def evaluation(items):
    """an Evaluation with these item scores and an attempt for each."""
    results = {item: attempt(item, s) for item, s in items.items()}
    return Evaluation(dict(items), results)


def individual(prompt, score, generation=0):
    items = {"a": score, "b": 1.0 - score, "c": round(score / 2, 3)}
    return Individual(prompt, evaluation(items), generation)
