"""Terminal-Bench task folders (originals or inject_bugs.py variants) as a
Dataset. Scoring is partial credit, not Terminal-Bench's binary pass/fail."""

import sys
from pathlib import Path

from promptevo.datasets.base import Dataset, ItemResult, RepairResult
from promptevo.datasets.terminal_bench.problems import format_problems
from promptevo.datasets.terminal_bench.task import Mode, Task
from promptevo.models import OllamaModel

DEFAULT_TASKS_DIR = Path(__file__).parents[4] / "data/raw/terminal-bench/tasks"


class TerminalBench(Dataset):
    """task folders named items under data_dir."""

    def __init__(
        self,
        items: list[str],
        data_dir: str | Path | None = None,
        mode: Mode = Mode.NORMAL,
    ):
        tasks_dir = Path(data_dir or DEFAULT_TASKS_DIR)
        self.tasks = [Task(tasks_dir / item) for item in items]
        self.mode = mode  # not NORMAL = calibration, the solver isn't asked

    def run(self, prompt: str, solver: OllamaModel) -> dict[str, ItemResult]:
        results = {}
        for task in self.tasks:
            run = task.run(self.mode, prompt, solver)
            if run.problems:
                why = format_problems(run.problems)
                print(f"{task.name}: {why}", file=sys.stderr)
            results[task.name] = RepairResult(
                tests=run.tests,
                problems=[str(p) for p in run.problems],
                to_fix=task.baseline.to_fix,
                passing=task.baseline.keep,
            )
        return results
