"""The study's runs and the transfer rescoring, with fake logs, data and
solver. The study's logs are written as raw JSON here, so these tests don't
lean on runlog.py to check runlog.py's readers."""

import json
import sys
from pathlib import Path

from conftest import RecordingModel, check_golden
from promptevo.study import design, transfer
from test_cli import FakeData


# folders and the python path differ between machines
def _portable(command):
    results = str(design.RESULTS.parent)
    return [
        "<python>" if part == sys.executable
        else part.replace(results, "<results>").replace("\\", "/")
        for part in command
    ]


def test_study_commands_and_setup():
    log = Path("run.jsonl")
    commands = [
        _portable(design.command(7, arm, "http://x", log))
        for arm in design.ARMS
    ]
    other = _portable(design.command(7, "aware-mixed", "http://x", log, "ds"))
    check_golden("study", {
        "commands": [*commands, other],
        "setup": design.setup({"root": "Qwen/Coder"}, design.SOLVER),
        "split": design.split(7),
    })


# one study unit, every arm logged with the same baselines and its own pick
def _fake_study(folder, unit):
    shared = {
        "best seed": "Seed A",
        "no prompt": "",
        "seed 0": "Seed A",
        "seed 1": "Seed B",
    }
    items = {"t1": {"partial": 1.0, "binary": 1.0, "naive": 1.0,
                    "output": "x"}}
    for arm in transfer.ARMS:
        prompts = shared | {"evolved": f"{arm} best", "val pick": f"{arm}!"}
        results = {
            name: {"prompt": p, "score": 1.0, "items": items}
            for name, p in prompts.items()
        }
        path = folder / f"unit{unit}" / f"{arm}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"event": "test", "results": results}) + "\n",
            encoding="utf-8",
        )
    split = {"test": ["t1", "t2", "t3"]}
    (folder / f"unit{unit}" / "split.json").write_text(json.dumps(split))


def test_transfer_rescores_the_study_prompts(tmp_path, monkeypatch):
    _fake_study(tmp_path / "study", 3)
    monkeypatch.setattr(transfer, "STUDY", tmp_path / "study")
    monkeypatch.setattr(transfer, "RESULTS", tmp_path / "transfer")
    monkeypatch.setattr(transfer, "HumanEval", FakeData)
    monkeypatch.setattr(
        transfer, "solver", lambda *a, **k: RecordingModel()
    )
    status = transfer._run(3, "fake:1b", "http://x")
    assert status.startswith("unit 3: done"), status
    log = tmp_path / "transfer" / "fake-1b" / "unit3.jsonl"
    events = [json.loads(line) for line in log.read_text().splitlines()]
    check_golden("transfer-unit", events)
