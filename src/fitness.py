"""fitness function for the EA. for each task:

1. build the task's docker images
2. start the task container, copy its code out
3. send the model the prompt, the task, the tests the code fails, and the code
4. write the files the model sends back
5. put the code in a test container, run the task's tests
6. score = share of the tests the original buggy code failed that now pass.
   0 if a test that used to pass now fails, or the tests didn't run.
   partial credit system is different from the original terminal bench binary!!!
   (we figure out which tests used to pass/fail by
    running the tests once on the untouched code)

fitness = average score over all tasks.

the Status enum class, Problem enum class, and fitness function are meant to be
the public facing API of this file. Our EA will use the fitness function and
we can use Status and Problem to make decisions about what broken code means for
our algorithm.
"""

import argparse
import ast
import functools
import json
import re
import subprocess
import sys
import tempfile
import random
import tomllib
import urllib.request
import urllib.error
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple
from ollama import chat

# terminal-bench task folders
DEFAULT_TASKS_DIR = Path(__file__).parents[1] / "data/raw/terminal-bench/tasks"

# "do not train on this" marker at the top of each instruction.md
# this is removed before sending the prompt to the model
CANARY_RE = re.compile(r"<!--[^>]*harbor-canary[^>]*-->\s*", re.IGNORECASE)

# also strip
_CANARY_LINE_RE = re.compile(
    r"^.*(harbor-canary|BENCHMARK DATA SHOULD NEVER).*\n?", re.MULTILINE
)



# enum for what went wrong in a run. empty means all fine
class Status(StrEnum):
    BAD_JSON = "bad json"           # usually output got cut off
    NO_FILES = "no files"
    UNKNOWN_PATH = "unknown path"
    SYNTAX_ERROR = "syntax error"
    TIMEOUT = "tests timed out"
    NO_RESULTS = "no test results"  # anti-cheat check failed or crash


# one thing that went wrong. path is the related file (if any)
class Problem(NamedTuple):
    status: Status
    path: str = ""


# one task's outcome for a prompt
class TaskResult(NamedTuple):
    score: float
    problems: list[Problem]  # empty = ran fine
    tests: dict[str, bool]   # {test name: passed?}. empty = tests didn't run


# what to do to the code before testing it
class _Mode(StrEnum):
    NORMAL = "normal"    # run the model
    PERFECT = "perfect"  # run the reference solution
    NOTHING = "nothing"  # leave the buggy code as is


# tests the untouched code fails / passes, and what the failures said
class _Baseline(NamedTuple):
    to_fix: frozenset[str]
    keep: frozenset[str]
    errors: dict[str, str]  # {to_fix test: its error}


def _format_problems(problems):
    return ", ".join(f"{p.status} {p.path}".strip() for p in problems) or "ok"


_OLLAMA = "http://localhost:11434/api/chat"
# ollama defaults to 4k ctx under 24GB vram, way too small for whole files.
# 16k fits fully on an 8GB gpu
# too small for batched-eval-parity so should be skipped locally
_NUM_CTX = 8192
# max output tokens. w/o this a model can loop until ctx is full (not good).
# cut off output = no files
_NUM_PREDICT = 4096
# greedy + fixed seed so the same prompt always gets the same score. with
# ollama's default temperature (0.8) one sample per task is too noisy for the
# EA to rank prompts by
_TEMPERATURE = 0
_SEED = 0

# only fixed text added to the prompt.
# this should be neutral and not bias the model to any particular style of code.
# fenced code, not json mode: in json the model has to escape every quote and
# newline of a whole file, and a 7b model gets that wrong often enough that
# syntax errors hide whether it found the bug
_FORMAT = """
Reply format: you may reason first. Then, for each file you change, write a line
`### <path>` with the exact path shown above, followed by the complete new
contents of that file in a ```python fenced code block. Only include files you change.
"""


# runs a docker command and returns its stdout
def _docker(*args):
    out = subprocess.run(
        ["docker", *args], check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


# prompts ollama
def _ask(model, system, user):
    body = {
        "model": model,
        "stream": False,
        "options": {
            "num_ctx": _NUM_CTX,
            "num_predict": _NUM_PREDICT,
            "temperature": _TEMPERATURE,
            "seed": _SEED,
        },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    req = urllib.request.Request(
        _OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    try:
        # a stuck request would hang the EA forever
        with urllib.request.urlopen(req, timeout=1200) as r:
            return json.load(r)["message"]["content"]
    except urllib.error.HTTPError as e:
        error_message = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama request failed (HTTP {e.code}): {error_message}"
        ) from e


# {container path: host path} for .py files under the copied artifacts
def _code_files(artifacts):
    files = {}
    for artifact, local in artifacts.items():
        for host in local.rglob("*.py"):
            if host.is_file():
                rel = host.relative_to(local).as_posix()
                files[artifact.rstrip("/") + "/" + rel] = host
    return dict(sorted(files.items()))


_FENCE_RE = re.compile(r"^```[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)


# [(path, code)] for each fenced block in the reply. the path is a shown one
# either in a comment on the block's first line, or the last one mentioned in
# the text since the previous block. models put the `### path` line in both
def _parse_reply(reply, paths):
    changes, prev_end = [], 0
    for m in _FENCE_RE.finditer(reply):
        code = m.group(1)
        first, _, rest = code.partition("\n")
        inside = [p for p in paths if p in first]
        if inside and first.lstrip().startswith("#"):
            path, code = max(inside, key=len), rest
        else:
            before = reply[prev_end : m.start()]
            seen = [(before.rfind(p), len(p), p) for p in paths if p in before]
            path = max(seen)[2] if seen else None
        prev_end = m.end()
        if path:
            changes.append((path, code))
    return changes


# writes model's files over the shown ones. returns problems
def _apply_reply(reply, files):
    changes = _parse_reply(reply, list(files))
    if not changes:
        return [Problem(Status.NO_FILES)]
    problems = []
    for path, code in changes:
        try:
            ast.parse(code)
        except SyntaxError as e:
            print(f"{path}: line {e.lineno}: {e.msg}\n{code}", file=sys.stderr)
            problems.append(Problem(Status.SYNTAX_ERROR, path))
        files[path].write_text(code, encoding="utf-8", newline="\n")
    return problems


# reads task.toml. raises on tasks we can't run
def _load_config(task_dir):
    cfg = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    compose = task_dir / "environment" / "docker-compose.yaml"
    if cfg.get("environment", {}).get("gpus") or compose.exists():
        raise ValueError(
            f"{task_dir.name}: needs gpu or docker-compose, not supported"
        )
    return cfg


# builds the task and test images if missing, returns their names.
# skips existing ones bc docker build hits docker hub every time, even cached,
# so a network blip would crash the EA. delete the images to rebuild
def _build_images(task_dir):
    task_img, test_img = f"tb-env-{task_dir.name}", f"tb-test-{task_dir.name}"
    for img, context in ((task_img, "environment"), (test_img, "tests")):
        exists = subprocess.run(
            ["docker", "image", "inspect", img], capture_output=True
        )
        if exists.returncode != 0:
            _docker("build", "-q", "-t", img, task_dir / context)
    return task_img, test_img


# runs the reference solve.sh in the container
def _run_solution(container, task_dir):
    _docker("cp", task_dir / "solution", f"{container}:/solution")
    _docker("exec", container, "bash", "/solution/solve.sh")


# copies artifact dirs out into tmp. returns {container path: host dir}
def _copy_out(container, artifact_paths, tmp):
    artifacts = {
        path: Path(tmp) / str(i) for i, path in enumerate(artifact_paths)
    }
    for path, local in artifacts.items():
        _docker("cp", f"{container}:{path.rstrip('/')}/.", local)
    return artifacts


# copies artifact dirs from the host into the container
def _copy_in(container, artifacts):
    for path, local in artifacts.items():
        _docker("exec", container, "mkdir", "-p", path)
        _docker("cp", f"{local}/.", f"{container}:{path}")


# the tests the untouched code fails, shown to the model like a ci log.
# without them the model has to guess which part of the code is broken
def _failure_report(errors):
    lines = ["The test suite currently reports these failures:", ""]
    for name, error in list(errors.items())[:8]:
        # tracebacks can be huge, keep the start of each
        lines += [f"- {name}", "  " + " ".join(error.split())[:600]]
    return "\n".join(lines)


# shows the model the task, failing tests and code, writes its reply.
# returns problems
def _ask_and_apply(task_dir, prompt, model, artifacts):
    files = _code_files(artifacts)
    raw = (task_dir / "instruction.md").read_text(encoding="utf-8")
    instruction = CANARY_RE.sub("", raw).strip()
    report = _failure_report(_baseline(task_dir).errors)
    shown = "\n\n".join(
        "### {}\n```python\n{}\n```".format(
            path, _CANARY_LINE_RE.sub("", host.read_text(encoding="utf-8"))
        )
        for path, host in files.items()
    )
    user = f"{instruction}\n\n{report}\n\n{shown}\n\n{_FORMAT}"
    reply = _ask(model, prompt, user)
    return _apply_reply(reply, files)


# runs the tests. returns ({test name: passed}, problems, {failed test: error})
def _run_tests(verifier, cfg):
    timeout = cfg.get("verifier", {}).get("timeout_sec", 600)
    try:
        subprocess.run(
            ["docker", "exec", verifier, "bash", "/tests/test.sh"],
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {}, [Problem(Status.TIMEOUT)], {}
    # per-test results, pytest writes these
    out = subprocess.run(
        ["docker", "exec", verifier, "cat", "/logs/verifier/ctrf.json"],
        capture_output=True,
        text=True,
    )
    try:
        tests = json.loads(out.stdout)["results"]["tests"]
    except (json.JSONDecodeError, KeyError):
        return {}, [Problem(Status.NO_RESULTS)], {}
    results = {t["name"]: t["status"] == "passed" for t in tests}
    # the traceback says where it failed, the message often says nothing
    errors = {
        t["name"]: t.get("trace") or t.get("message") or ""
        for t in tests
        if t["status"] != "passed"
    }
    return results, [], errors


def _run_task(task_dir, prompt, model, mode=_Mode.NORMAL):
    task_dir = Path(task_dir)
    cfg = _load_config(task_dir)
    task_img, test_img = _build_images(task_dir)

    task_container = verifier = None
    problems = []
    try:
        task_container = _docker("run", "-d", task_img, "sleep", "infinity")
        if mode == _Mode.PERFECT:
            _run_solution(task_container, task_dir)

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = _copy_out(task_container, cfg["artifacts"], tmp)
            if mode == _Mode.NORMAL:
                problems = _ask_and_apply(task_dir, prompt, model, artifacts)
            verifier = _docker("run", "-d", test_img, "sleep", "infinity")
            _copy_in(verifier, artifacts)

        results, test_problems, errors = _run_tests(verifier, cfg)
        return results, problems + test_problems, errors
    finally:
        for c in (task_container, verifier):
            if c:
                subprocess.run(["docker", "rm", "-f", c], capture_output=True)


# what tests pass/fail without any code changes. runs once per task and cached
@functools.cache
def _baseline(task_dir):
    results, problems, errors = _run_task(task_dir, "", "", _Mode.NOTHING)
    to_fix = frozenset(t for t, ok in results.items() if not ok)
    if not to_fix:
        why = _format_problems(problems)
        raise RuntimeError(f"{task_dir}: untouched code fails no tests ({why})")
    keep = frozenset(t for t, ok in results.items() if ok)
    return _Baseline(to_fix, keep, errors)


# share of to_fix now passing. 0 if a keep test broke or no results
def _score(results, base):
    if not results or not all(results.get(t) for t in base.keep):
        return 0.0
    return sum(bool(results.get(t)) for t in base.to_fix) / len(base.to_fix)

def mutate(genome, mutation_rate, model): # mutate random letters
    if random.random() >= mutation_rate:
        return genome
    # construct mutation instruction
    mutation_instruction = f"""
    You are mutating a system prompt for a coding assistant. 

    You are to make EXACTLY one change that will improve the model's performance on the tasks provided.

    Perserve only the original purpose. 
    Return ONLY the modified prompt

    Original prompt:
    {genome}
    """
    body = chat(
        model= model,
        stream=False,
        messages=[
            {"role": "user", "content": mutation_instruction},
        ],
    )
    return body.message.content.strip()

def crossover(parent_a, parent_b, model):
    # Construct an instruction containing both parents
    instruction = f"""
    You are performing a crossover operation for a system prompt for a coding assistant. 

    The prompt shuld retain its original intent.

    You are to combine the information from both parent prompts to create a new child prompt.

    Parent A:
    {parent_a}

    Parent B:
    {parent_b}

    Return ONLY the new child prompt.
    """
    # Send both parents to the LLM
    body = chat(
            model= model,
            stream=False,
            messages=[
                {"role": "user", "content": instruction},
            ],
        )
    # Extract the generated child prompt
    child_prompt = body.message.content.strip()
    # Return the child prompt
    return child_prompt


def run_evolution(tasks, mu, lambda_, generations, mutation_rate, model, seed,  plus=True,):
    #initial prompts (genome)
    initial_prompts = [
        "You are an expert level software engineer. Solve the coding task carefully, following all directions.",
        "You are a coding assistant. Make sure you thoroughly analyze the question before solving.",
        "You are an experienced programmer. Make the least amount of changes to produce a more efficient solution, while preserving functionality.",
        "You are a meticulous software engineer. Consider edge cases, and make sure your solution is correct."
    ]
    parents = initial_prompts[:mu]

    random.seed(seed)
    if mu != len(parents):
        raise ValueError("mu must be equal to the number of initial parents.")
    if mu < 2:
        raise ValueError("crossover needs mu of at least 2.")
    if not plus and lambda_ < mu:
        raise ValueError("lambda_ must be at least mu when plus is False.")
    scores = {}
    for parent in parents:
        scores[parent] = fitness(parent, tasks, model)[0]
    # generate children
    for generation in range(generations):
        children = []
        for _ in range(lambda_):
            # two different parents. a prompt crossed with itself is just
            # a paraphrase of it
            parent_a, parent_b = random.sample(parents, 2)
            child = crossover(parent_a, parent_b, model)
            child = mutate(child, mutation_rate, model)
            children.append(child)
        # skip over repeats
        for child in children:
            if child not in scores:
                scores[child] = fitness(child, tasks, model)[0]
        pool = parents + children if plus else children
        # select best mu parents for next gen
        parents = sorted(
            pool, 
            key=lambda genome: scores[genome], 
            reverse=True)[:mu]
    best_prompt = max(scores, key=scores.get)
    return best_prompt, scores[best_prompt]
    
        
   


#------------------------------------------------------------------------------#
# Fitness evaluation function for our evolutionary algorithm
# Returns the mean score of prompt over tasks,
# Also returns {task: TaskResult} (per-task score, problems, test results)
# this second return value might be useful for some mutation experiments
# if we only want to find the fitness score then only use first return value:
#     `score, _ = fitness(prompt, tasks, model)`
def fitness(
    prompt: str,
    tasks: list[str],
    model: str,
    tasks_dir: str | Path = DEFAULT_TASKS_DIR,
) -> tuple[float, dict[str, TaskResult]]:
    by_task = {}
    for t in tasks:
        task_dir = Path(tasks_dir) / t
        tests, problems, _ = _run_task(task_dir, prompt, model)
        if problems:
            print(f"{t}: {_format_problems(problems)}", file=sys.stderr)
        score = _score(tests, _baseline(task_dir))
        by_task[t] = TaskResult(score, problems, tests)
    mean = sum(r.score for r in by_task.values()) / len(tasks)
    return mean, by_task

#------------------------------------------------------------------------------#

# for testing
if __name__ == "__main__":
    best_prompt, best_score = run_evolution(
    tasks=["risk-scorer-replay"],
    mu=2,
    lambda_=1,
    generations=2,
    mutation_rate=0.5,
    model="qwen2.5-coder:7b",
    seed=42,
    plus=True
    )

    print("\nBest Prompt:", best_prompt)
    print("Best Score:", best_score)
    raise SystemExit()

    
    genome = "You are an expert software engineer. Solve coding problems carefully"
    mutated = mutate(
        genome = genome,
        mutation_rate=1.0,
        model="qwen2.5-coder:7b"
    )
    print(f"Original: {genome}")
    print(f"Mutated: {mutated}")

    #initial prompts (genome)
    initial_prompts = [
        "You are an expert level software engineer. Solve the coding task carefully, following all directions.",
        "You are a coding assistant. Make sure you thoroughly analyze the question before solving.",
        "You are an experienced programmer. Make the least amount of changes to produce a more efficient solution, while preserving functionality.",
        "You are a meticulous software engineer. Consider edge cases, and make sure your solution is correct."
    ]

    parentA = initial_prompts[0]
    parentB = initial_prompts[1]
    child = crossover(
        parent_a=parentA,
        parent_b=parentB,
        model="qwen2.5-coder:7b"
    )
    print(f"ParentA: {parentA}")
    print(f"ParentB: {parentB}")
    print(f"Child: {child}")

    parser = argparse.ArgumentParser()



    # task folder names under --tasks-dir
    parser.add_argument("tasks", nargs="+")

    # what is to be evolved. this default one is for testing
    parser.add_argument("--prompt", default="Do the coding.")

    # ollama model tag
    parser.add_argument("--model", default="qwen2.5-coder:7b")

    # see _Mode
    parser.add_argument(
        "--mode", type=_Mode, choices=list(_Mode), default=_Mode.NORMAL
    )

    # where the task folders are
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)

    args = parser.parse_args()

    for t in args.tasks:
        task_dir = Path(args.tasks_dir) / t
        results, problems, _ = _run_task(
            task_dir, args.prompt, args.model, args.mode
        )
        score = _score(results, _baseline(task_dir))
        print(t, score, _format_problems(problems))
