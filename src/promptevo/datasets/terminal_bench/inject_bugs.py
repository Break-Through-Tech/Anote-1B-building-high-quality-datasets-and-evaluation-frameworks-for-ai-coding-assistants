"""Bug-injected variants of Terminal-Bench tasks. For each task:

1. Run the reference solution in the task container, copy the fixed code out
2. Mutate one token in the fixed code (comparison, arithmetic, and/or,
   True/False, int constant, min/max). Token-level so comments and formatting
   stay the same and nothing points the model at the changed file
3. Write the variant as a normal task folder. Its environment is the original
   task image + one layer with the mutated code, so it runs like any other
   Task
4. Keep the variant only if the task's own tests catch the bug

Variants go to data/processed/bug-injected/tasks, with manifest.json next
to it describing each injected bug. Use with
    TerminalBench(variant_names, VARIANTS_DIR)
or --data-dir in the CLI. From src/:
    python -m promptevo.datasets.terminal_bench.inject_bugs risk-scorer-replay

Only supports tasks whose solution changes Python files (solve.sh copies .py
files into the artifact dirs). PERFECT mode doesn't work on variants because
the overlay files are root owned and solve.sh runs as the task's user.
"""

import argparse
import ast
import io
import json
import random
import re
import shutil
import tempfile
import tokenize
from pathlib import Path
from typing import NamedTuple

from promptevo.datasets.terminal_bench.dataset import DEFAULT_TASKS_DIR
from promptevo.datasets.terminal_bench.docker import Container
from promptevo.datasets.terminal_bench.problems import format_problems
from promptevo.datasets.terminal_bench.task import (
    CANARY_RE,
    Mode,
    Task,
    code_files,
)

DEFAULT_OUT = Path(__file__).parents[4] / "data/processed/bug-injected"

# regression first. the original task's symptoms are already fixed, so it's
# only background
INSTRUCTION = """\
A recent change to this code introduced a regression. Before that change the
code passed all of its tests; now some of them fail. Find the regression and
fix it with the smallest change you can. Do not rewrite code that is not
related to the failure.

For context, below is the original task this code was written for. The
problems it describes have already been fixed and are not the current bug,
but its rules (for example, files that must not be modified) still apply.

---

{original}
"""

_SWAPS = {
    "cmp": {"<": "<=", "<=": "<", ">": ">=", ">=": ">", "==": "!=", "!=": "=="},
    "arith": {
        "+": "-", "-": "+", "*": "/", "/": "*", "//": "/",
        "+=": "-=", "-=": "+=",
    },
    "bool": {"and": "or", "or": "and", "True": "False", "False": "True"},
    "minmax": {"min": "max", "max": "min"},
}


# one place a single token can be swapped. row is 1-based, col 0-based
class _Site(NamedTuple):
    path: str
    row:  int
    col:  int
    old:  str
    new:  str
    kind: str


class BugInjector:
    """writes caught variants of tasks into out_dir/tasks."""

    def __init__(
        self,
        out_dir: str | Path = DEFAULT_OUT,
        per_task: int = 6,
        max_tries: int = 18,
        seed: int = 0,
    ):
        self.tasks_dir = Path(out_dir) / "tasks"
        self.per_task = per_task
        self.max_tries = max_tries
        self.rng = random.Random(seed)

    def inject(self, task: Task) -> list[dict]:
        """makes up to per_task caught variants of one task. returns manifest
        entries."""
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        fixed = _fixed_code(task)

        # unmutated code has to pass everything or the overlay is wrong
        control = self.tasks_dir / f"{task.name}--control"
        _write_variant(control, task, fixed)
        run = Task(control).run(Mode.NOTHING)
        _remove(control)
        if not run.tests or not all(run.tests.values()):
            why = format_problems(run.problems)
            raise RuntimeError(f"{task.name}: fixed code fails tests ({why})")

        sites = [s for path, code in fixed.items() for s in _sites(path, code)]
        self.rng.shuffle(sites)
        entries = []
        tries = 0
        for site in sites:
            if len(entries) == self.per_task or tries == self.max_tries:
                break
            mutated = _mutate(fixed[site.path], site)
            if mutated is None:
                continue
            tries += 1
            name = f"{task.name}--bug{tries:02d}"
            variant = self.tasks_dir / name
            _write_variant(variant, task, {**fixed, site.path: mutated})
            try:
                base = Task(variant).baseline
            except RuntimeError:  # tests don't catch it
                print(f"{name}: survived {site.kind} {site.path}:{site.row}")
                _remove(variant)
                continue
            line = mutated.splitlines()[site.row - 1].strip()
            print(
                f"{name}: caught by {len(base.to_fix)} tests  "
                f"{site.path}:{site.row}  {line}"
            )
            entries.append({
                "name": name,
                "task": task.name,
                "kind": site.kind,
                "path": site.path,
                "line": site.row,
                "original": site.old,
                "mutated": site.new,
                "mutated_line": line,
                "to_fix": sorted(base.to_fix),
                "keep": sorted(base.keep),
            })
        return entries

    # just instruction.md, no image rebuild needed
    def refresh_instruction(self, name: str, task: Task):
        _write_instruction(self.tasks_dir / name, task)


# every token in code we know how to mutate
def _sites(path, code):
    tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    sites = []
    for i, tok in enumerate(tokens):
        (row, col), s = tok.start, tok.string
        # mutating the main guard just stops the program, too easy
        if "__name__" in tok.line:
            continue
        nxt = tokens[i + 1].string if i + 1 < len(tokens) else ""
        if tok.type == tokenize.OP:
            for kind in ("cmp", "arith"):
                if s in _SWAPS[kind]:
                    new = _SWAPS[kind][s]
                    sites.append(_Site(path, row, col, s, new, kind))
        elif tok.type == tokenize.NAME:
            if s in _SWAPS["bool"]:
                new = _SWAPS["bool"][s]
                sites.append(_Site(path, row, col, s, new, "bool"))
            elif s in _SWAPS["minmax"] and nxt == "(":
                new = _SWAPS["minmax"][s]
                sites.append(_Site(path, row, col, s, new, "minmax"))
        elif tok.type == tokenize.NUMBER and re.fullmatch(r"\d+", s):
            sites.append(_Site(path, row, col, s, str(int(s) + 1), "const"))
    return sites


# code with the site's token swapped. None if that isn't valid python
def _mutate(code, site):
    lines = code.splitlines(keepends=True)
    line = lines[site.row - 1]
    end = site.col + len(site.old)
    if line[site.col : end] != site.old:
        return None  # splitlines and tokenize disagree on line breaks
    lines[site.row - 1] = line[: site.col] + site.new + line[end:]
    mutated = "".join(lines)
    try:
        if ast.dump(ast.parse(mutated)) == ast.dump(ast.parse(code)):
            return None
    except SyntaxError:  # e.g. *args becomes /args
        return None
    return mutated


# {container path: code} after the reference solution runs
def _fixed_code(task):
    task.build_images()
    with (
        Container(task.task_image) as box,
        tempfile.TemporaryDirectory() as tmp,
    ):
        task.run_solution(box)
        artifacts = task.copy_out(box, tmp)
        return {
            path: host.read_text(encoding="utf-8")
            for path, host in code_files(artifacts).items()
        }


# canary stays on top so Task.instruction still strips it
def _write_instruction(out, task):
    raw = (task.path / "instruction.md").read_text(encoding="utf-8")
    canary = CANARY_RE.match(raw.lstrip())
    text = (canary.group(0) if canary else "") + INSTRUCTION.format(
        original=task.instruction
    )
    (out / "instruction.md").write_text(text, encoding="utf-8")


# writes a variant task folder. code is {container path: code}
def _write_variant(out, task, code):
    if out.exists():
        shutil.rmtree(out)
    for sub in ("tests", "solution"):
        shutil.copytree(task.path / sub, out / sub)
    shutil.copy(task.path / "task.toml", out / "task.toml")
    _write_instruction(out, task)
    env = out / "environment"
    for path, text in code.items():
        dst = env / "overlay" / path.lstrip("/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8", newline="\n")
    (env / "Dockerfile").write_text(
        f"FROM {task.task_image}\nCOPY overlay/ /\n", encoding="utf-8"
    )


def _remove(variant_dir):
    Task(variant_dir).remove_images()
    shutil.rmtree(variant_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # task folder names under --tasks-dir
    parser.add_argument("tasks", nargs="+")

    # caught variants to keep per task
    parser.add_argument("--per-task", type=int, default=6)

    # mutants to test per task before giving up. each one runs the tests once
    parser.add_argument("--max-tries", type=int, default=18)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)

    # rewrite instruction.md of the variants in the manifest, nothing else
    parser.add_argument("--refresh-instructions", action="store_true")
    args = parser.parse_args()

    injector = BugInjector(args.out, args.per_task, args.max_tries, args.seed)
    manifest_path = args.out / "manifest.json"
    if args.refresh_instructions:
        manifest = json.loads(manifest_path.read_text())
        for v in manifest:
            if v["task"] in args.tasks:
                task = Task(args.tasks_dir / v["task"])
                injector.refresh_instruction(v["name"], task)
        raise SystemExit(f"refreshed instructions under {injector.tasks_dir}")

    manifest = []
    for t in args.tasks:
        manifest += injector.inject(Task(args.tasks_dir / t))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{len(manifest)} variants -> {injector.tasks_dir}")
