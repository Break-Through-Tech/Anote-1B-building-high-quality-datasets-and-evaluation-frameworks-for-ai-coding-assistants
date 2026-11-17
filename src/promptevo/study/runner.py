"""Runs the study (design.py). From src/:

    python -m promptevo.study --units 0 1 2 3

Runs go a few at a time against one vLLM server, and a run whose log
already has its test scores is skipped, so running it again picks up where
it stopped. Needs the vLLM server from the README, Ollama for the mutator
and Docker.
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from promptevo import paths
from promptevo.llm.models import served_model
from promptevo.search.runlog import LoggedRun
from promptevo.study.design import (
    ARMS,
    MAIN_ARMS,
    MUTATOR,
    SOLVER,
    command,
    results,
    setup,
    split,
)


def keep_awake():
    """stops windows from sleeping until the process ends."""
    if sys.platform == "win32":
        import ctypes
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)


def _parse_args():
    parser = argparse.ArgumentParser(prog="promptevo.study")
    parser.add_argument("--units", type=int, nargs="+", required=True)
    parser.add_argument(
        "--arms", nargs="+", choices=list(ARMS), default=MAIN_ARMS
    )
    parser.add_argument(
        "--parallel", type=int, default=4,
        help="runs at once. they share the solver, the mutator takes turns",
    )
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument(
        "--solver", default=SOLVER,
        help="the vllm --served-model-name to evolve prompts for. one that "
        "isn't the default logs to its own results folder",
    )
    return parser.parse_args()


# everything a run needs, checked before starting hours of them. returns
# what the vllm server says it serves
def _preflight(url, solver):
    try:
        with httpx.Client(timeout=10) as client:
            model = served_model(client, url, solver)
    except ValueError as e:
        raise SystemExit(e)
    subprocess.run(["docker", "info"], capture_output=True, check=True)
    models = httpx.get("http://localhost:11434/api/tags", timeout=10).json()
    if MUTATOR not in [m["name"] for m in models["models"]]:
        raise SystemExit(f"ollama doesn't have {MUTATOR}")
    return model


# the setup, next to the results. a later run with a different setup stops
# here, so two setups never end up in one study. new arms can be added, an
# arm that already ran can't change
def _manifest(model, solver):
    wanted = setup(model, solver)
    path = results(solver) / "study.json"
    if path.exists():
        _check_same(path, wanted)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=paths.SRC,
        capture_output=True,
        text=True,
    ).stdout.strip()
    # the commit is only for reference, the code may have uncommitted edits
    path.write_text(json.dumps({**wanted, "commit": commit}, indent=1))


def _check_same(path, wanted):
    old = json.loads(path.read_text())
    old.pop("commit", None)
    old_arms = old.pop("arms")
    changed = [
        name for name, arm in old_arms.items()
        if wanted["arms"].get(name) != arm
    ]
    rest = {k: v for k, v in wanted.items() if k != "arms"}
    if changed or old != rest:
        raise SystemExit(
            f"{path} has a different setup, see the file. "
            f"changed arms: {changed}"
        )


def _run(unit, arm, url, solver):
    folder = results(solver) / f"unit{unit}"
    log = folder / f"{arm}.jsonl"
    if LoggedRun(log).finished:
        return f"unit {unit} {arm}: already done"
    if log.exists():
        # a run that died partway. the log only gets appended to, so move
        # it out of the way instead of mixing two runs in one file
        log.rename(log.with_suffix(f".incomplete-{int(time.time())}"))
    start = time.time()
    with (folder / f"{arm}.out").open("w") as out, \
            (folder / f"{arm}.err").open("w") as err:
        code = subprocess.run(
            command(unit, arm, url, log, solver),
            cwd=paths.SRC,
            stdout=out,
            stderr=err,
        ).returncode
    minutes = (time.time() - start) / 60
    if code or not LoggedRun(log).finished:
        return f"unit {unit} {arm}: FAILED, exit {code}, {minutes:.0f} min"
    return f"unit {unit} {arm}: done in {minutes:.0f} min"


def main():
    args = _parse_args()
    keep_awake()
    home = results(args.solver)
    home.mkdir(parents=True, exist_ok=True)
    _manifest(_preflight(args.vllm_url, args.solver), args.solver)
    for unit in args.units:
        folder = home / f"unit{unit}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "split.json").write_text(json.dumps(split(unit), indent=1))
    # unit by unit, so whole units finish first for the paired comparison
    runs = [(unit, arm) for unit in args.units for arm in args.arms]
    with ThreadPoolExecutor(args.parallel) as pool:
        jobs = [pool.submit(_run, unit, arm, args.vllm_url, args.solver)
                for unit, arm in runs]
        for job in as_completed(jobs):
            print(job.result(), flush=True)
