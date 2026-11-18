"""Scores one prompt on some tasks, for testing. From src/:

    python -m promptevo.unused.terminal_bench risk-scorer-replay \\
        --mode perfect

Perfect should score 1, nothing 0.
"""

import argparse
from pathlib import Path

from promptevo.llm.models import solver
from promptevo.scoring.evaluation import partial
from promptevo.unused.terminal_bench import (
    DEFAULT_TASKS_DIR,
    Mode,
    TerminalBench,
)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tasks", nargs="+", help="task folder names under --tasks-dir"
    )
    parser.add_argument(
        "--prompt", default="Do the coding.",
        help="the system prompt to score",
    )
    parser.add_argument("--model", default="qwen2.5-coder:7b",
                        help="ollama model tag")
    parser.add_argument(
        "--mode", type=Mode, choices=list(Mode), default=Mode.NORMAL,
        help="perfect runs the reference solution, nothing leaves the bug",
    )
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    return parser.parse_args()


def main():
    args = _parse_args()
    dataset = TerminalBench(args.tasks, args.tasks_dir, args.mode)
    for name, result in dataset.run(args.prompt, solver(args.model)).items():
        print(name, partial(result), ", ".join(result.problems) or "ok")


if __name__ == "__main__":
    main()
