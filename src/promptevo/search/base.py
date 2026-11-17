"""What the two searches have in common. A search turns seed prompts into
prompts ranked best first. The EA is in evolution.py, GEPA's loop in
unused/gepa.py."""

import sys
from abc import ABC, abstractmethod

from promptevo.scoring.evaluation import Evaluator
from promptevo.scoring.individual import Individual
from promptevo.search.operators import Operators
from promptevo.search.runlog import RunLog


class Search(ABC):
    """a prompt search. run() also keeps the seeds as scored, for the
    comparison on the test set."""

    def __init__(self, operators: Operators, log: RunLog | None = None):
        self.operators = operators
        self.log       = log or RunLog()
        self.seeds: list[Individual] = []

    @abstractmethod
    def run(self, seeds: list[str]) -> list[Individual]:
        """searches from seeds. returns the prompts it ended with, best
        first."""

    @property
    @abstractmethod
    def rollouts(self) -> int:
        """solver calls so far."""

    # a crash scores 0 instead of ending a long run
    def _scored(self, evaluator: Evaluator, prompt, generation):
        ind = Individual(prompt, evaluator.try_evaluate(prompt), generation)
        self.log.scored(ind)
        return ind

    # the operator's text, or fallback if it threw
    def _safely(self, operator, fallback):
        try:
            return operator().strip()
        except Exception as e:
            print(f"operator failed: {e!r}", file=sys.stderr)
            return fallback

    def _new_population(self, generation, population):
        self.operators.observe(population)
        self.log.generation(
            generation,
            population,
            self.rollouts,
            self.operators.mutator_calls,
        )
