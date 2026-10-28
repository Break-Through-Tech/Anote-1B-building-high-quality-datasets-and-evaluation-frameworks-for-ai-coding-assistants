"""Types shared by the EA, operators and evaluator."""

from typing import NamedTuple


# score is what selection ranks by, cases are per-test scores for lexicase
class Evaluation(NamedTuple):
    score:    float
    cases:    dict[str, float]
    feedback: str = ""      # shown to the mutator
    crashed:  bool = False  # evaluate threw


class Individual(NamedTuple):
    prompt:     str
    evaluation: Evaluation
    generation: int  # 0 = seed
