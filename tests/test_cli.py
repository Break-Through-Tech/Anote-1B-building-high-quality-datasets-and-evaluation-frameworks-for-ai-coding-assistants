"""The whole CLI run (python -m promptevo) with a fake dataset and models."""

import contextlib
import hashlib
import io
import sys

import pytest

from conftest import RecordingModel, check_golden
from promptevo import __main__ as cli
from promptevo.datasets.base import Dataset, RepairResult
from promptevo.search.runlog import LoggedRun


class FakeData(Dataset):
    description = "Fix it."

    def __init__(self, items, data_dir=None, **options):
        self.items = list(items)

    @property
    def names(self):
        return self.items

    # sends the solver every item, then fixed test results per prompt
    def run(self, prompt, solver, items=None):
        chosen = [i for i in self.items if items is None or i in items]
        replies = solver.chat_many([f"fix {i}" for i in chosen], prompt)
        out = {}
        for item, reply in zip(chosen, replies):
            h = hashlib.sha256(f"{prompt}/{item}".encode()).digest()
            out[item] = RepairResult(
                tests={"t1": h[0] % 2 == 0, "t2": h[1] % 3 == 0},
                inputs=f"fix {item}",
                output=reply,
                to_fix=frozenset({"t1", "t2"}),
            )
        return out


RUNS = {
    "ea": [
        "a", "b", "c", "--dataset", "fake", "--validation", "v1", "v2",
        "--held-out", "t1", "t2", "--mu", "3", "--lambda", "3",
        "--generations", "3", "--crossover-rate", "0.5", "--mutations",
        "rewrite", "eda", "zero-order-hyper", "first-order", "--seed", "5",
        "--drift-check", "--screen", "2",
    ],
    "gepa": [
        "a", "b", "c", "d", "--dataset", "fake", "--search", "gepa",
        "--budget", "60", "--validation", "v1", "v2", "--held-out", "t1",
        "--mutations", "reflective", "lamarckian",
    ],
}


@pytest.mark.parametrize("name", list(RUNS))
def test_cli_run(name, monkeypatch, tmp_path):
    monkeypatch.setitem(cli.DATASETS, "fake", FakeData)
    monkeypatch.setattr(cli, "solver", lambda *a, **k: RecordingModel())
    monkeypatch.setattr(cli, "mutator", lambda name: RecordingModel())
    log = tmp_path / "run.jsonl"
    monkeypatch.setattr(
        sys, "argv", ["promptevo", *RUNS[name], "--log", str(log)]
    )
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cli.main()
    printed = out.getvalue().replace(str(log), "<log>")
    check_golden(f"cli-{name}", {
        "printed": printed,
        "log": LoggedRun(log).events,
    })
