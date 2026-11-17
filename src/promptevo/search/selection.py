"""Parent selection. The study picks parents uniformly, the other selectors
are in unused/selection.py."""

import random
from abc import ABC, abstractmethod

from promptevo.scoring.individual import Individual


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
