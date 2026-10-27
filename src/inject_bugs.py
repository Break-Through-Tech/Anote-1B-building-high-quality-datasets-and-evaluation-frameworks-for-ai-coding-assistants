"""bug-injected variants of terminal-bench tasks. for each task:

1. run the reference solution in the task container, copy the fixed code out
2. mutate one token in the fixed code (comparison, arithmetic, and/or,
   True/False, int constant, min/max). token-level so comments and formatting
   stay the same and nothing points the model at the changed file
3. write the variant as a normal task folder. its environment is the original
   task image + one layer with the mutated code, so fitness.py runs it as is
4. keep the variant only if the task's own tests catch the bug

variants go to data/processed/bug-injected/tasks, with manifest.json next to
it describing each injected bug. use with
    fitness(prompt, variant_names, model, tasks_dir=VARIANTS_DIR)

only supports tasks whose solution changes python files (solve.sh copies .py
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
import subprocess
import tempfile
import tokenize
from pathlib import Path

import fitness as f

DEFAULT_OUT = Path(__file__).parents[1] / "data/processed/bug-injected"

# variant instruction. the regression comes first, the original task is only
# background: its listed symptoms were fixed by the reference solution, and a
# model that reads them as the current bug goes hunting for the wrong thing
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
    "arith": {"+": "-", "-": "+", "*": "/", "/": "*", "//": "/",
              "+=": "-=", "-=": "+="},
    "bool": {"and": "or", "or": "and", "True": "False", "False": "True"},
    "minmax": {"min": "max", "max": "min"},
}


# one place a single token can be swapped. row is 1-based, col 0-based
class _Site:
    def __init__(self, path, row, col, old, new, kind):
        self.path, self.row, self.col = path, row, col
        self.old, self.new, self.kind = old, new, kind


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
                    sites.append(_Site(path, row, col, s, _SWAPS[kind][s], kind))
        elif tok.type == tokenize.NAME:
            if s in _SWAPS["bool"]:
                sites.append(_Site(path, row, col, s, _SWAPS["bool"][s], "bool"))
            elif s in _SWAPS["minmax"] and nxt == "(":
                sites.append(_Site(path, row, col, s, _SWAPS["minmax"][s], "minmax"))
        elif tok.type == tokenize.NUMBER and re.fullmatch(r"\d+", s):
            sites.append(_Site(path, row, col, s, str(int(s) + 1), "const"))
    return sites


# code with the site's token swapped. None if that isn't valid python
def _mutate(code, site):
    lines = code.splitlines(keepends=True)
    line = lines[site.row - 1]
    if line[site.col: site.col + len(site.old)] != site.old:
        return None  # splitlines and tokenize disagree on line breaks
    lines[site.row - 1] = line[: site.col] + site.new + line[site.col + len(site.old):]
    mutated = "".join(lines)
    try:
        if ast.dump(ast.parse(mutated)) == ast.dump(ast.parse(code)):
            return None
    except SyntaxError:  # e.g. *args -> /args
        return None
    return mutated


# runs the reference solution and copies the fixed code out.
# returns {container path: code}
def _fixed_code(task_dir):
    cfg = f._load_config(task_dir)
    task_img, _ = f._build_images(task_dir)
    container = f._docker("run", "-d", task_img, "sleep", "infinity")
    try:
        f._run_solution(container, task_dir)
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = f._copy_out(container, cfg["artifacts"], tmp)
            return {
                path: host.read_text(encoding="utf-8")
                for path, host in f._code_files(artifacts).items()
            }
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)


# the canary stays at the top so fitness.py strips it like on the original
def _write_instruction(out, task_dir):
    raw = (task_dir / "instruction.md").read_text(encoding="utf-8")
    canary = f.CANARY_RE.match(raw.lstrip())
    original = f.CANARY_RE.sub("", raw).strip()
    text = (canary.group(0) if canary else "") + INSTRUCTION.format(original=original)
    (out / "instruction.md").write_text(text, encoding="utf-8")


# writes a variant task folder. code is {container path: code}
def _write_variant(out, task_dir, code):
    if out.exists():
        shutil.rmtree(out)
    for sub in ("tests", "solution"):
        shutil.copytree(task_dir / sub, out / sub)
    shutil.copy(task_dir / "task.toml", out / "task.toml")
    _write_instruction(out, task_dir)
    env = out / "environment"
    for path, text in code.items():
        dst = env / "overlay" / path.lstrip("/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8", newline="\n")
    (env / "Dockerfile").write_text(
        f"FROM tb-env-{task_dir.name}\nCOPY overlay/ /\n", encoding="utf-8"
    )


def _remove(variant_dir):
    name = variant_dir.name
    subprocess.run(
        ["docker", "rmi", "-f", f"tb-env-{name}", f"tb-test-{name}"],
        capture_output=True,
    )
    shutil.rmtree(variant_dir, ignore_errors=True)


# makes up to per_task caught variants of one task. returns manifest entries
def inject(task_dir, out_dir, per_task, max_tries, rng):
    task_dir = Path(task_dir)
    fixed = _fixed_code(task_dir)

    # the unmutated fixed code has to pass every test, or the overlay is wrong
    control = out_dir / f"{task_dir.name}--control"
    _write_variant(control, task_dir, fixed)
    results, problems, _ = f._run_task(control, "", "", f._Mode.NOTHING)
    _remove(control)
    if not results or not all(results.values()):
        why = f._format_problems(problems)
        raise RuntimeError(f"{task_dir.name}: fixed code fails tests ({why})")

    sites = [s for path, code in fixed.items() for s in _sites(path, code)]
    rng.shuffle(sites)
    entries = []
    tries = 0
    for site in sites:
        if len(entries) == per_task or tries == max_tries:
            break
        mutated = _mutate(fixed[site.path], site)
        if mutated is None:
            continue
        tries += 1
        name = f"{task_dir.name}--bug{tries:02d}"
        variant = out_dir / name
        _write_variant(variant, task_dir, {**fixed, site.path: mutated})
        try:
            base = f._baseline(variant)
        except RuntimeError:  # tests don't catch it
            print(f"{name}: survived {site.kind} {site.path}:{site.row}")
            _remove(variant)
            continue
        line = mutated.splitlines()[site.row - 1].strip()
        print(f"{name}: caught by {len(base.to_fix)} tests  {site.path}:{site.row}  {line}")
        entries.append({
            "name": name,
            "task": task_dir.name,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # task folder names under --tasks-dir
    parser.add_argument("tasks", nargs="+")

    # caught variants to keep per task
    parser.add_argument("--per-task", type=int, default=6)

    # mutants to test per task before giving up. each one runs the tests once
    parser.add_argument("--max-tries", type=int, default=18)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks-dir", type=Path, default=f.DEFAULT_TASKS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)

    # rewrite instruction.md of the variants in the manifest, nothing else.
    # no image rebuild needed, fitness.py reads it from the host
    parser.add_argument("--refresh-instructions", action="store_true")
    args = parser.parse_args()

    if args.refresh_instructions:
        manifest = json.loads((args.out / "manifest.json").read_text())
        for v in manifest:
            if v["task"] in args.tasks:
                _write_instruction(args.out / "tasks" / v["name"], args.tasks_dir / v["task"])
        raise SystemExit(f"refreshed instructions under {args.out / 'tasks'}")

    out_tasks = args.out / "tasks"
    out_tasks.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    manifest = []
    for t in args.tasks:
        manifest += inject(
            args.tasks_dir / t, out_tasks, args.per_task, args.max_tries, rng
        )
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"{len(manifest)} variants -> {out_tasks}")
