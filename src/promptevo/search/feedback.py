"""What the mutator is told about a parent's evaluation. The run log keeps
the same summary for every scored prompt."""

from enum import StrEnum

from promptevo.datasets.base import ItemResult
from promptevo.scoring.evaluation import Evaluation
from promptevo.search.templates import FAILURES_SHOWN, SCORE_SHOWN


# what the rewrite mutation sees
class Feedback(StrEnum):
    NONE     = "none"      # blind rewrite
    SCORE    = "score"     # score only
    FAILURES = "failures"  # score + failing tests


def describe(evaluation: Evaluation, feedback: Feedback) -> str:
    """the part of the rewrite request about the parent's evaluation."""
    if feedback == Feedback.NONE:
        return ""
    text = SCORE_SHOWN.format(score=evaluation.score)
    report = summary(evaluation)
    if feedback == Feedback.FAILURES and report:
        text += FAILURES_SHOWN.format(feedback=report)
    return text


# no error text, but a test's name can be the whole assert (humaneval),
# expected value included
def summary(evaluation: Evaluation) -> str:
    """a line per item with its score and what still fails, or the error
    if evaluating threw."""
    if evaluation.crashed:
        return f"evaluation failed: {evaluation.error}"
    lines = []
    for item, result in evaluation.results.items():
        line = f"- {item}: score {evaluation.items[item]:.2f}"
        failing = _failing(result)
        if failing:
            line += f", still failing: {', '.join(failing[:8])}"
        if result.problems:
            line += f", problems: {', '.join(result.problems)}"
        lines.append(line)
    return "\n".join(lines)


def item_feedback(result: ItemResult, show_tests: bool = True) -> str:
    """what went wrong on one item, in a line or two. show_tests=False
    leaves out the failing tests themselves, which for humaneval are the
    asserts with their expected values."""
    failing = _failing(result)
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


def _failing(result):
    return [test for test, passed in result.tests.items() if not passed]
