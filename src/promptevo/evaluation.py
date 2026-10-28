"""Turns dataset results into an Evaluation for the EA. A signal turns one
item's results into a score (partial or binary)."""

from enum import StrEnum

from promptevo.datasets.base import Dataset, ItemResult
from promptevo.models import OllamaModel
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


# which item score the ea optimizes
class Signal(StrEnum):
    PARTIAL = "partial"
    BINARY  = "binary"

    def score(self, result: ItemResult) -> float:
        """this signal's score for one item."""
        return binary(result) if self == Signal.BINARY else partial(result)


class Evaluator:
    """scores a prompt. runs the dataset with the solver, averages items."""

    def __init__(
        self,
        dataset: Dataset,
        solver: OllamaModel,
        signal: Signal = Signal.PARTIAL,
    ):
        self.dataset = dataset
        self.solver  = solver
        self.signal  = signal

    def evaluate(self, prompt: str) -> Evaluation:
        results = self.dataset.run(prompt, self.solver)
        scores = {item: self.signal.score(r) for item, r in results.items()}
        return Evaluation(
            score=sum(scores.values()) / len(scores),
            cases=_cases(results, scores),
            feedback=_feedback(results, scores),
        )


# per-test cases for lexicase. items that scored 0 get all 0s
def _cases(results, scores):
    return {
        f"{item}::{test}": float(passed and scores[item] > 0)
        for item, result in results.items()
        for test, passed in result.tests.items()
    }


# for the FAILURES mutator. test names only, error text could leak
# expected values
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
