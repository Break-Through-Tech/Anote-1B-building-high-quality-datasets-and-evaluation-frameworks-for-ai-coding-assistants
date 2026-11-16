"""Types shared by the EA, operators and evaluator."""

from typing import NamedTuple


# score is what selection ranks by, cases are per-test scores for lexicase
class Evaluation(NamedTuple):
    score:    float
    cases:    dict[str, float]
    feedback: str = ""      # shown to the mutator
    crashed:  bool = False  # evaluate threw
    # {item: score}, for pareto selection and screening
    items:    dict[str, float] | None = None
    # {item: ItemResult} with the solver's replies. not logged, too big
    results:  dict | None = None


class Individual(NamedTuple):
    prompt:     str
    evaluation: Evaluation
    generation: int  # 0 = seed
