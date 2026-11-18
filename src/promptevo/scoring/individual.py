"""A prompt with its evaluation, what the searches keep and select from."""

from typing import NamedTuple

from promptevo.scoring.evaluation import Evaluation


class Individual(NamedTuple):
    prompt:     str
    evaluation: Evaluation
    generation: int  # 0 = seed. for gepa, the step
