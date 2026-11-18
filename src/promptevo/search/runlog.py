"""The JSONL log of a run (https://jsonlines.org/), and reading it back.

A run appends one event per line: every scored prompt, mutation and
generation, then the validation and test-set scores at the end. RunLog
writes them, LoggedRun reads them back, so no other module needs to know
the format.
"""

import json
from functools import cached_property
from pathlib import Path
from typing import NamedTuple

from promptevo.scoring.evaluation import Evaluation, binary, naive, partial
from promptevo.scoring.individual import Individual
from promptevo.search.feedback import summary


class RunLog:
    """jsonl log, appended right away so a crash doesn't lose it. without a
    path it keeps nothing, for runs that don't need a log."""

    def __init__(self, path: str | Path | None = None):
        self.path = None if path is None else Path(path)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def scored(self, ind: Individual):
        evaluation = ind.evaluation
        self._write({
            "event": "scored",
            "generation": ind.generation,
            "prompt": ind.prompt,
            "score": evaluation.score,
            # null for a crash, as in the study's logs
            "items": None if evaluation.crashed else evaluation.items,
            "cases": evaluation.cases,
            "feedback": summary(evaluation),
            "crashed": evaluation.crashed,
        })

    def generation(
        self,
        generation: int,
        population: list[Individual],
        rollouts: int | None = None,
        mutator_calls: int | None = None,
    ):
        self._write({
            "event": "generation",
            "generation": generation,
            "survivors": [
                {"prompt": p.prompt, "score": p.evaluation.score}
                for p in population
            ],
            "rollouts": rollouts,
            "mutator_calls": mutator_calls,
        })

    def screened(
        self, prompt: str, generation: int, score: float, parent: float
    ):
        self._write({
            "event": "screened",
            "generation": generation,
            "prompt": prompt,
            "score": score,
            "parent_score": parent,
        })

    def mutation(self, kind: str, parent: str, child: str):
        self._write({
            "event": "mutation",
            "kind": kind,
            "parent": parent,
            "child": child,
        })

    def validation(self, scores: dict[str, float]):
        self._write({
            "event": "validation",
            "results": [
                {"prompt": prompt, "score": score}
                for prompt, score in scores.items()
            ],
        })

    def drift(self, prompt: str, before: float, after: float):
        self._write({
            "event": "drift",
            "prompt": prompt,
            "before": before,
            "after": after,
        })

    def test(
        self, prompts: dict[str, str], evaluations: dict[str, Evaluation]
    ):
        """the test set, {name: prompt} and {name: its evaluation}. every
        item's scores and reply go in too."""
        record = {
            name: {
                "prompt": prompts[name],
                "score": evaluations[name].score,
                "items": _items(evaluations[name]),
            }
            for name in prompts
        }
        self._write({"event": "test", "results": record})

    def _write(self, record):
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# every signal for every item, so a report can use any of them later
def _items(evaluation):
    return {
        item: {
            "partial": partial(result),
            "binary": binary(result),
            "naive": naive(result),
            "output": result.output,
        }
        for item, result in evaluation.results.items()
    }


class Tested(NamedTuple):
    """one prompt's test set result, read back from a log."""

    prompt: str
    score:  float  # mean partial credit
    # {item: {"partial", "binary", "naive", "output"}}
    items:  dict[str, dict]

    @property
    def strict(self) -> float:
        """share of items with every assert passing."""
        passed = [item["binary"] for item in self.items.values()]
        return sum(passed) / len(passed)


class Mutated(NamedTuple):
    """one mutation, read back from a log."""

    kind:   str
    parent: str
    child:  str  # empty when the mutator's reply was dropped


# the test event is the last thing a run writes
_TEST = '"event": "test"'


class LoggedRun:
    """a run's log, read back. a log that's still being written only counts
    as finished once it has the test set."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @property
    def finished(self) -> bool:
        """True once the run got as far as scoring the test set."""
        if not self.path.exists():
            return False
        with self.path.open(encoding="utf-8") as f:
            return any(_TEST in line for line in f)

    @cached_property
    def events(self) -> list[dict]:
        """every event in the log, in order."""
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    # only parses the test line, the rest of a study log is large
    @cached_property
    def test(self) -> dict[str, Tested]:
        """{name: Tested}, the test set of a finished run."""
        with self.path.open(encoding="utf-8") as f:
            record = next(
                json.loads(line)["results"] for line in f if _TEST in line
            )
        return {
            name: Tested(r["prompt"], r["score"], r.get("items", {}))
            for name, r in record.items()
        }

    @property
    def tried(self) -> int:
        """prompts the search scored after the seeds. a mutation that
        returns its parent, nothing, or too much text adds none."""
        return sum(
            e["event"] == "scored" and e["generation"] > 0
            for e in self.events
        )

    @property
    def crashed(self) -> int:
        """prompts scored 0 because the solver or sandbox failed, not the
        prompt."""
        return sum(
            e["event"] == "scored" and e.get("crashed", False)
            for e in self.events
        )

    @property
    def mutations(self) -> list[Mutated]:
        return [
            Mutated(e["kind"], e["parent"], e["child"])
            for e in self.events
            if e["event"] == "mutation"
        ]
