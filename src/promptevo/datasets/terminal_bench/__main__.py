"""Scores one prompt on some tasks, for testing. From src/:

    python -m promptevo.datasets.terminal_bench risk-scorer-replay \\
        --mode perfect

Perfect should score 1, nothing 0.
"""

import argparse
from pathlib import Path

from promptevo.datasets.terminal_bench import (
    DEFAULT_TASKS_DIR,
    Mode,
    TerminalBench,
)
from promptevo.evaluation import partial
from promptevo.models import solver

parser = argparse.ArgumentParser()

# task folder names under --tasks-dir
parser.add_argument("tasks", nargs="+")

# what is to be evolved. this default one is for testing
parser.add_argument("--prompt", default="Do the coding.")

# ollama model tag
parser.add_argument("--model", default="qwen2.5-coder:7b")

# see Mode
parser.add_argument(
    "--mode", type=Mode, choices=list(Mode), default=Mode.NORMAL
)

# where the task folders are
parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)

args = parser.parse_args()

dataset = TerminalBench(args.tasks, args.tasks_dir, args.mode)
for name, result in dataset.run(args.prompt, solver(args.model)).items():
    print(name, partial(result), ", ".join(result.problems) or "ok")
