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


# evoprompt's ga (sec 3.2). chance of being picked is proportional to score
class RouletteSelector(Selector):
    def pick(self, pool, rng):
        weights = [ind.evaluation.score for ind in pool]
        if not any(weights):
            return rng.choice(pool)
        return rng.choices(pool, weights)[0]


# gepa's candidate selection (alg. 2, as in gepa_utils.py of
# github.com/gepa-ai/gepa). every item has a front, the prompts with its best
# score. lowest scores first, a prompt is dropped while every front it's on
# still has another prompt. picked in proportion to how many fronts it's on
class ParetoSelector(Selector):
    def pick(self, pool, rng):
        scores = [ind.evaluation.items or {} for ind in pool]
        items = sorted({i for s in scores for i in s})
        fronts = []
        for i in items:
            best = max(s.get(i, 0.0) for s in scores)
            fronts.append({
                k for k, s in enumerate(scores) if s.get(i, 0.0) == best
            })
        alive = set().union(*fronts)
        for k in sorted(alive, key=lambda k: pool[k].evaluation.score):
            if all(len(f & alive) > 1 for f in fronts if k in f):
                alive.discard(k)
        weights = [
            sum(k in f for f in fronts) if k in alive else 0
            for k in range(len(pool))
        ]
        if not any(weights):
            return rng.choice(pool)
        return rng.choices(pool, weights)[0]


# always the best average score. gepa's SelectBestCandidate ablation (sec 3.1)
class GreedySelector(Selector):
    def pick(self, pool, rng):
        best = max(ind.evaluation.score for ind in pool)
        return rng.choice([i for i in pool if i.evaluation.score == best])


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


# lexicase on a random share of the test cases for every pick. in genetic
# programming this keeps more variety and overfits less
# (https://link.springer.com/article/10.1007/s42979-025-04281-x)
class DownsampledLexicaseSelector(LexicaseSelector):
    def __init__(self, rate: float = 0.25):
        self.rate = rate

    def pick(self, pool, rng):
        cases = sorted({c for ind in pool for c in ind.evaluation.cases})
        if not cases:
            return rng.choice(pool)
        kept = set(rng.sample(cases, max(1, round(len(cases) * self.rate))))
        trimmed = [
            ind._replace(evaluation=ind.evaluation._replace(cases={
                c: s for c, s in ind.evaluation.cases.items() if c in kept
            }))
            for ind in pool
        ]
        picked = super().pick(trimmed, rng)
        return pool[trimmed.index(picked)]
