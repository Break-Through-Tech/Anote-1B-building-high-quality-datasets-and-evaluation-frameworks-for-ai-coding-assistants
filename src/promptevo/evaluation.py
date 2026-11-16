"""Turns dataset results into an Evaluation for the EA. A signal turns one
item's results into a score (partial, binary or naive)."""

from enum import StrEnum

from promptevo.datasets.base import Dataset, ItemResult
from promptevo.models import ChatModel
from promptevo.types import Evaluation


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


class Evaluator:
    """scores a prompt. runs the dataset with the solver, averages items."""

    def __init__(
        self,
        dataset: Dataset,
        solver: ChatModel,
        signal: Signal = Signal.PARTIAL,
    ):
        self.dataset = dataset
        self.solver  = solver
        self.signal  = signal
        # solver attempts so far, the budget gepa counts in
        self.rollouts = 0
        self._cache = {}  # {prompt: {item: ItemResult}}
        self._names = None  # every item, known after the first full run

    def evaluate(
        self, prompt: str, items: list[str] | None = None
    ) -> Evaluation:
        """scores prompt on every item, or just items. each (prompt, item)
        only ever runs once."""
        known = self._cache.setdefault(prompt, {})
        wanted = items if items is not None else self._names
        missing = None if wanted is None else [
            i for i in wanted if i not in known
        ]
        if missing is None or missing:
            try:
                fresh = self.dataset.run(prompt, self.solver, missing)
            except Exception:
                # the solver already ran even if the tests then crashed.
                # missing is None on the first full run, that's every item
                self.rollouts += len(
                    self.dataset if missing is None else missing
                )
                raise
            self.rollouts += len(fresh)
            known.update(fresh)
            if self._names is None and items is None:
                self._names = list(fresh)
        results = {i: known[i] for i in (wanted or list(known))}
        scores = {item: self.signal.score(r) for item, r in results.items()}
        return Evaluation(
            score=sum(scores.values()) / len(scores),
            cases=_cases(results, scores),
            feedback=_feedback(results, scores),
            items=scores,
            results=results,
        )


def item_feedback(result: ItemResult, show_tests: bool = True) -> str:
    """what went wrong on one item, in a line or two. show_tests=False
    leaves out the failing tests themselves, which for humaneval are the
    asserts with their expected values."""
    failing = [t for t, passed in result.tests.items() if not passed]
    if not result.tests:
        text = "The tests didn't run."
    elif failing:
        text = f"{len(failing)} of {len(result.tests)} tests failed"
        text += (": " + "; ".join(failing[:3])) if show_tests else "."
    else:
        text = "Every test passed."
    if result.problems:
        text += f" Problems: {', '.join(result.problems)}."
    return text


# per-test cases for lexicase. items that scored 0 get all 0s
def _cases(results, scores):
    return {
        f"{item}::{test}": float(passed and scores[item] > 0)
        for item, result in results.items()
        for test, passed in result.tests.items()
    }


# for the FAILURES mutator. no error text, but a test's name can be the
# whole assert (humaneval), expected value included
def _feedback(results, scores):
    lines = []
    for item, result in results.items():
        line = f"- {item}: score {scores[item]:.2f}"
        failing = [t for t, passed in result.tests.items() if not passed]
        if failing:
            line += f", still failing: {', '.join(failing[:8])}"
        if result.problems:
            line += f", problems: {', '.join(result.problems)}"
        lines.append(line)
    return "\n".join(lines)
