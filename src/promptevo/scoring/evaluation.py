"""Scores prompts. A signal turns one item's results into a score (partial,
binary or naive), and an Evaluation is a prompt's item scores together with
the attempts behind them."""

import sys
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property

from promptevo.datasets.base import Dataset, ItemResult
from promptevo.llm.models import ChatModel


# share of target tests passing, 0 if a keep test broke. apps "test case
# average", swe-bench style for repair
def partial(result: ItemResult) -> float:
    """partial credit for one item, 0 to 1."""
    tests, target = result.tests, result.target
    if not tests or not target:
        return 0.0
    if not all(tests.get(t) for t in result.keep):
        return 0.0
    return sum(bool(tests.get(t)) for t in target) / len(target)


# apps "strict accuracy"
def binary(result: ItemResult) -> float:
    """1 only if every target test passes and nothing broke."""
    return float(partial(result) == 1.0)


# share of every test passing, the ones the bug doesn't touch included.
# partial's baseline. sending the buggy code back unchanged already scores
# well here
def naive(result: ItemResult) -> float:
    """naive partial credit for one item, 0 to 1."""
    if not result.tests:
        return 0.0
    return sum(map(bool, result.tests.values())) / len(result.tests)


# which item score the ea optimizes
class Signal(StrEnum):
    PARTIAL = "partial"
    BINARY  = "binary"
    NAIVE   = "naive"

    def score(self, result: ItemResult) -> float:
        """this signal's score for one item."""
        return _SIGNALS[self](result)


_SIGNALS = {
    Signal.PARTIAL: partial,
    Signal.BINARY: binary,
    Signal.NAIVE: naive,
}


# frozen because the evaluator hands the same results out again. score and
# cases are worked out from the items, so they can't disagree with them
@dataclass(frozen=True)
class Evaluation:
    """a prompt's scores on some items, and the attempts behind them."""

    items:   dict[str, float]       # {item: signal score}
    results: dict[str, ItemResult]  # not logged, too big
    error:   str | None = None      # what evaluating threw, nothing ran then

    @classmethod
    def failed(cls, error: Exception) -> "Evaluation":
        """an evaluation that threw. it scores 0."""
        return cls({}, {}, repr(error))

    @property
    def crashed(self) -> bool:
        return self.error is not None

    # cached, selection reads these again and again
    @cached_property
    def score(self) -> float:
        """mean item score, what selection ranks by."""
        if not self.items:
            return 0.0
        return sum(self.items.values()) / len(self.items)

    # tests of an item that scored 0 count as failed
    @cached_property
    def cases(self) -> dict[str, float]:
        """{"item::test": 0 or 1}, the per-test scores lexicase uses."""
        return {
            f"{item}::{test}": float(passed and self.items[item] > 0)
            for item, result in self.results.items()
            for test, passed in result.tests.items()
        }


class Evaluator:
    """scores prompts on a dataset with the solver. each (prompt, item) only
    ever runs once."""

    def __init__(
        self,
        dataset: Dataset,
        solver: ChatModel,
        signal: Signal = Signal.PARTIAL,
    ):
        self.dataset  = dataset
        self.solver   = solver
        self.signal   = signal
        self.rollouts = 0   # solver attempts so far, the budget counts these
        self._results = {}  # {prompt: {item: ItemResult}}

    def evaluate(
        self, prompt: str, items: list[str] | None = None
    ) -> Evaluation:
        """scores prompt on every item, or just items."""
        wanted = self.dataset.names if items is None else items
        known = self._results.setdefault(prompt, {})
        missing = [item for item in wanted if item not in known]
        if missing:
            known.update(self._run(prompt, missing))
        results = {item: known[item] for item in wanted}
        scores = {item: self.signal.score(r) for item, r in results.items()}
        return Evaluation(scores, results)

    def try_evaluate(
        self, prompt: str, items: list[str] | None = None
    ) -> Evaluation:
        """evaluate(), but a crash scores 0 instead of ending a long run."""
        try:
            return self.evaluate(prompt, items)
        except Exception as e:
            print(f"evaluation failed: {e!r}", file=sys.stderr)
            return Evaluation.failed(e)

    # the solver already ran even if the tests then crashed, so those count
    def _run(self, prompt, items):
        try:
            fresh = self.dataset.run(prompt, self.solver, items)
        except Exception:
            self.rollouts += len(items)
            raise
        self.rollouts += len(fresh)
        return fresh
