"""Parent selection."""

import random
from abc import ABC, abstractmethod

from promptevo.types import Individual


class Selector(ABC):
    """picks parents. subclasses only need pick()."""

    @abstractmethod
    def pick(self, pool: list[Individual], rng: random.Random) -> Individual:
        ...

    def parents(
        self, population: list[Individual], rng: random.Random
    ) -> tuple[Individual, Individual]:
        """two parents with different prompts if there are two."""
        # dupes can leave just 1 unique prompt
        if len(population) < 2:
            return population[0], population[0]
        a = self.pick(population, rng)
        rest = [p for p in population if p.prompt != a.prompt]
        return a, self.pick(rest, rng)


class UniformSelector(Selector):
    def pick(self, pool, rng):
        return rng.choice(pool)


# see https://lexicase.ai/ for a short description of what this means
class LexicaseSelector(Selector):
    def pick(self, pool, rng):
        cases = sorted({c for ind in pool for c in ind.evaluation.cases})
        rng.shuffle(cases)
        for case in cases:
            best = max(ind.evaluation.cases.get(case, 0.0) for ind in pool)
            pool = [
                i for i in pool if i.evaluation.cases.get(case, 0.0) == best
            ]
            if len(pool) == 1:
                break
        return rng.choice(pool)
