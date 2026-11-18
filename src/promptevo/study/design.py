"""The study's design: how each unit splits HumanEvalFix, the solver and
mutator, the arms, and the command line every run gets. runner.py runs it,
report.py and transfer.py read its results.

Every unit is its own random split of the 164 problems into train,
validation and test. Every arm runs once per unit with the same split and
seed, so arms get compared inside a unit (a paired design).
"""

import json
import random
import re
import sys
from pathlib import Path
from typing import NamedTuple

from promptevo import paths

RESULTS = paths.RESULTS / "study"

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


# study.json keeps these as written, so an arm that already ran can't change
class Arm(NamedTuple):
    signal:     str  # fitness the ea selects on
    # rewrite is blind, reflective sees attempts. a tuple of several gets
    # one picked at random per child
    mutation:   str | tuple[str, ...]
    hide_tests: bool  # reflective without the failing asserts

    @property
    def mutations(self) -> list[str]:
        if isinstance(self.mutation, str):
            return [self.mutation]
        return list(self.mutation)


ARMS = {
    "naive-blind":     Arm("naive", "rewrite", False),
    "aware-blind":     Arm("partial", "rewrite", False),
    "naive-reflect":   Arm("naive", "reflective", True),
    "aware-reflect":   Arm("partial", "reflective", True),
    # the failing asserts hold the expected values, so this is the arm that
    # can leak tests into the prompt
    "aware-reflect-tests": Arm("partial", "reflective", False),
    # not in the main grid, only runs with --arms
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


def safe_name(model: str) -> str:
    """model name as a folder name, windows doesn't allow colons."""
    return re.sub(r"[^\w.-]", "-", model)


def results(solver: str) -> Path:
    """where solver's study lives. another solver gets its own folder, so
    its runs never mix with the study solver's."""
    if solver == SOLVER:
        return RESULTS
    return RESULTS.parent / f"study-{safe_name(solver)}"


def command(
    unit: int, arm: str, url: str, log: Path, solver: str = SOLVER
) -> list[str]:
    """the promptevo call for one run."""
    sets = split(unit)
    chosen = ARMS[arm]
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
        "--signal", chosen.signal,
        "--mutations", *chosen.mutations,
        *(["--hide-tests"] if chosen.hide_tests else []),
        *EA_FLAGS,
        "--seed", str(unit),
        "--log", str(log),
    ]


def setup(model: dict, solver: str) -> dict:
    """what study.json records. model is what vllm serves under solver."""
    # through json so tuples compare equal to the lists read back
    arms = json.loads(json.dumps(
        {name: arm._asdict() for name, arm in ARMS.items()}
    ))
    return {
        "solver": solver,
        "solver_weights": model["root"],
        "mutator": MUTATOR,
        "train": TRAIN,
        "validation": VALIDATION,
        "ea_flags": EA_FLAGS,
        "arms": arms,
    }
