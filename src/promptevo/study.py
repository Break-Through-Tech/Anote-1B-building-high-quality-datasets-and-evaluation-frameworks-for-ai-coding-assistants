"""The fitness x feedback study on HumanEvalFix. From src/:

    python -m promptevo.study --units 0 1 2 3

Every unit is its own random split of the 164 problems into train,
validation and test. Every arm runs once per unit with the same split and
seed, so arms get compared inside a unit (a paired design). Runs go a few at
a time against one vLLM server, and a run whose log already has its test
scores is skipped, so running it again picks up where it stopped.

Needs the vLLM server from the README, Ollama for the mutator and Docker.
"""

import argparse
import json
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import httpx

_SRC = Path(__file__).parents[1]
RESULTS = _SRC.parent / "results" / "study"

# humanevalfix task ids
PROBLEMS = [f"Python/{i}" for i in range(164)]
TRAIN, VALIDATION = 40, 30  # the other 94 are the test set

SOLVER  = "qwen2.5-coder:1.5b"
MUTATOR = "qwen2.5-coder:7b"

# the same search for every arm. 4 seeds, then 8 generations of 4 children,
# each a mutated copy of one parent (no crossover, so the mutation is the
# only thing making children). 32 mutations and up to 36 x 40 train calls
EA_FLAGS = [
    "--mu", "4", "--lambda", "4", "--generations", "8",
    "--crossover-rate", "0", "--selection", "uniform",
]


# the operators aware-mixed picks from
_MIXED = ("rewrite", "first-order", "eda", "zero-order-hyper")


class Arm(NamedTuple):
    signal: str  # fitness the ea selects on
    # rewrite is blind, reflective sees attempts. a tuple of several gets
    # one picked at random per child
    mutation: str | tuple[str, ...]
    hide_tests: bool  # reflective without the failing asserts


ARMS = {
    "naive-blind":     Arm("naive", "rewrite", False),
    "aware-blind":     Arm("partial", "rewrite", False),
    "naive-reflect":   Arm("naive", "reflective", True),
    "aware-reflect":   Arm("partial", "reflective", True),
    # the failing asserts hold the expected values, so this is the arm that
    # can leak tests into the prompt
    "aware-reflect-tests": Arm("partial", "reflective", False),
    # not in the main grid, run it with --arms if there's time
    "binary-blind":    Arm("binary", "rewrite", False),
    # blind, but most children come from promptbreeder operators that can
    # change the prompt's whole strategy instead of one phrase. the others
    # mostly echoed their template or copied a train problem in a pilot
    "aware-mixed": Arm("partial", _MIXED, False),
    # its operators alone (rewrite alone is aware-blind) and with one left
    # out, to see which ones do the work
    **{f"aware-{op}": Arm("partial", op, False) for op in _MIXED[1:]},
    **{
        f"mixed-no-{op}": Arm(
            "partial", tuple(o for o in _MIXED if o != op), False
        )
        for op in _MIXED
    },
}
MAIN_ARMS = list(ARMS)[:5]


def split(unit: int) -> dict[str, list[str]]:
    """unit's train, validation and test problems. the same unit always
    gets the same split."""
    problems = list(PROBLEMS)
    random.Random(1000 + unit).shuffle(problems)
    return {
        "train": problems[:TRAIN],
        "validation": problems[TRAIN:TRAIN + VALIDATION],
        "test": problems[TRAIN + VALIDATION:],
    }


def results(solver: str) -> Path:
    """where solver's study lives. another solver gets its own folder, so
    its runs never mix with the study solver's."""
    if solver == SOLVER:
        return RESULTS
    return RESULTS.parent / ("study-" + re.sub(r"[^\w.-]", "-", solver))


def finished(log: Path) -> bool:
    """True once the run got as far as scoring the test set."""
    if not log.exists():
        return False
    with log.open(encoding="utf-8") as f:
        return any('"event": "test"' in line for line in f)


def command(
    unit: int, arm: str, url: str, log: Path, solver: str = SOLVER
) -> list[str]:
    """the promptevo call for one run."""
    sets = split(unit)
    signal, mutation, hide_tests = ARMS[arm]
    mutations = [mutation] if isinstance(mutation, str) else list(mutation)
    return [
        sys.executable, "-m", "promptevo", *sets["train"],
        "--validation", *sets["validation"],
        "--held-out", *sets["test"],
        "--dataset", "humaneval",
        "--solver", solver, "--vllm-url", url,
        "--cache", str(results(solver) / "replies.sqlite"),
        # test results don't depend on who wrote the code, so every solver
        # shares one
        "--sandbox-cache", str(RESULTS / "sandbox.sqlite"),
        "--mutator", MUTATOR,
        "--signal", signal,
        "--mutations", *mutations,
        *(["--hide-tests"] if hide_tests else []),
        *EA_FLAGS,
        "--seed", str(unit),
        "--log", str(log),
    ]


def _parse_args():
    parser = argparse.ArgumentParser(prog="promptevo.study")
    parser.add_argument("--units", type=int, nargs="+", required=True)
    parser.add_argument(
        "--arms", nargs="+", choices=list(ARMS), default=MAIN_ARMS
    )
    # runs at once. they share the solver, the mutator takes turns
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    # the vllm --served-model-name to evolve prompts for. one that isn't
    # SOLVER logs to results(solver)
    parser.add_argument("--solver", default=SOLVER)
    return parser.parse_args()


# everything a run needs, checked before starting hours of them. returns
# what the vllm server says it serves
def _preflight(url, solver):
    served = httpx.get(url + "/v1/models", timeout=10).json()["data"]
    model = next((m for m in served if m["id"] == solver), None)
    if model is None:
        raise SystemExit(f"vllm at {url} doesn't serve {solver}")
    subprocess.run(["docker", "info"], capture_output=True, check=True)
    models = httpx.get("http://localhost:11434/api/tags", timeout=10).json()
    if MUTATOR not in [m["name"] for m in models["models"]]:
        raise SystemExit(f"ollama doesn't have {MUTATOR}")
    return model


# the setup, next to the results. a later run with a different setup stops
# here, so two setups never end up in one study. new arms can be added, an
# arm that already ran can't change
def _manifest(model, solver):
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_SRC, capture_output=True, text=True
    ).stdout.strip()
    # through json so tuples compare equal to the lists read back
    arms = json.loads(json.dumps(
        {name: arm._asdict() for name, arm in ARMS.items()}
    ))
    setup = {
        "solver": solver,
        "solver_weights": model["root"],
        "mutator": MUTATOR,
        "train": TRAIN,
        "validation": VALIDATION,
        "ea_flags": EA_FLAGS,
        "arms": arms,
    }
    path = results(solver) / "study.json"
    if path.exists():
        old = json.loads(path.read_text())
        old.pop("commit", None)
        old_arms = old.pop("arms")
        changed = [
            name for name, arm in old_arms.items() if arms.get(name) != arm
        ]
        if changed or old != {k: v for k, v in setup.items() if k != "arms"}:
            raise SystemExit(
                f"{path} has a different setup, see the file. "
                f"changed arms: {changed}"
            )
    # the commit is only for reference, the code may have uncommitted edits
    path.write_text(json.dumps({**setup, "commit": commit}, indent=1))


def _run(unit, arm, url, solver):
    folder = results(solver) / f"unit{unit}"
    log = folder / f"{arm}.jsonl"
    if finished(log):
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
            cwd=_SRC,
            stdout=out,
            stderr=err,
        ).returncode
    minutes = (time.time() - start) / 60
    if code or not finished(log):
        return f"unit {unit} {arm}: FAILED, exit {code}, {minutes:.0f} min"
    return f"unit {unit} {arm}: done in {minutes:.0f} min"


if __name__ == "__main__":
    args = _parse_args()
    if sys.platform == "win32":
        # no sleeping while this runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED).
        # ends with the process
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
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
