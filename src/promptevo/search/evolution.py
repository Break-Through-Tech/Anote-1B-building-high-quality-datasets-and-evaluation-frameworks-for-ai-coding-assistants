"""(mu+lambda) prompt evolution. The evaluator, operators and selector get
passed in.

Per generation:
1. Pick 2 parents (Selector), or 1 when the child skips crossover
2. Crossover, maybe mutate. Without crossover, always mutate
3. Score new children. With screening, only the ones that beat their parent
   on a few train items first
4. Keep the best mu of parents + children (just children with comma).
   Ties are random

Logs every scored prompt and generation (runlog.py).
"""

import random
from typing import NamedTuple

from promptevo.scoring.evaluation import Evaluator
from promptevo.scoring.individual import Individual
from promptevo.search.base import Search
from promptevo.search.operators import Operators
from promptevo.search.runlog import RunLog
from promptevo.search.selection import Selector, UniformSelector


class EvolutionConfig(NamedTuple):
    mu: int
    lambda_: int
    generations: int
    mutation_rate: float = 0.5
    plus: bool = True  # False = comma
    seed: int = 0
    # longer operator output gets dropped, stops prompt bloat
    max_prompt_chars: int = 2000
    # chance a child starts with crossover. otherwise it's a mutated copy of
    # one parent, like gepa and promptbreeder mostly do
    crossover_rate: float = 1.0
    # gepa's minibatch check. above 0, a child gets scored on this many
    # random train items first and only fully if it beats its parent there
    screen: int = 0
    # stop early once the solver has been called this many times, so
    # methods can be compared at the same cost. 0 = no limit
    budget: int = 0


class Evolution(Search):
    """the ea loop"""

    def __init__(
        self,
        config: EvolutionConfig,
        evaluator: Evaluator,
        operators: Operators,
        selector: Selector | None = None,
        log: RunLog | None = None,
    ):
        super().__init__(operators, log)
        self.config    = config
        self.evaluator = evaluator
        self.selector  = selector or UniformSelector()

    @property
    def rollouts(self):
        return self.evaluator.rollouts

    def run(self, seeds: list[str]) -> list[Individual]:
        """runs the ea. returns final population, sorted by best"""
        self._check(len(seeds))
        self._rng = random.Random(self.config.seed)
        self._known = {}  # {prompt: Individual}, so none is scored twice

        self.seeds = [self._score(p, 0) for p in seeds]
        if all(ind.evaluation.crashed for ind in self.seeds):
            raise RuntimeError(
                "every seed evaluation crashed :(, check docker/ollama!"
            )
        population = self._survivors(self.seeds)
        self._new_population(0, population)
        for generation in range(1, self.config.generations + 1):
            if self._spent():
                break
            population = self._next_generation(population, generation)
        return population

    def _next_generation(self, population, generation):
        children = []
        for _ in range(self.config.lambda_):
            if self._spent():
                break
            prompt, parent = self._child(population)
            child = self._screened(prompt, parent, generation)
            if child is not None:
                children.append(child)
        pool = population + children if self.config.plus else children
        population = self._survivors(pool)
        self._new_population(generation, population)
        return population

    # checked before every child, so a run goes over by at most one
    # child's full score
    def _spent(self):
        budget = self.config.budget
        return bool(budget) and self.rollouts >= budget

    def _check(self, n_seeds):
        config = self.config
        if config.mu < 2:
            raise ValueError("crossover needs mu >= 2")
        if n_seeds < config.mu:
            raise ValueError(f"need at least mu={config.mu} seed prompts")
        if not config.plus and config.lambda_ < config.mu:
            raise ValueError("comma selection needs lambda_ >= mu")

    def _score(self, prompt, generation):
        if prompt not in self._known:
            self._known[prompt] = self._scored(
                self.evaluator, prompt, generation
            )
        return self._known[prompt]

    # crossover + maybe mutate, or just mutate one parent. bad output falls
    # back to the input. returns the child and the parent it's judged against
    def _child(self, population):
        ops = self.operators
        config = self.config
        # no rng draw at the default rate, so older runs replay the same
        if config.crossover_rate >= 1 or (
            self._rng.random() < config.crossover_rate
        ):
            a, b = self.selector.parents(population, self._rng)
            child = self._operate(
                lambda: ops.crossover(a.prompt, b.prompt), a.prompt
            )
            mutate = self._rng.random() < config.mutation_rate
        else:
            a = self.selector.pick(population, self._rng)
            child, mutate = a.prompt, True
        if mutate:
            # child has no score yet, use parent a's
            child = self._operate(
                lambda: ops.mutate(child, a.evaluation), child
            )
        return child, a

    # a child that doesn't beat its parent on a few random train items is
    # dropped without a full score. returns None for dropped ones
    def _screened(self, prompt, parent, generation):
        size = self.config.screen
        items = parent.evaluation.items
        if not size or prompt in self._known or not items:
            return self._score(prompt, generation)
        batch = self._rng.sample(sorted(items), min(size, len(items)))
        screen = self.evaluator.try_evaluate(prompt, batch)
        if screen.crashed:
            return None
        before = sum(items[i] for i in batch) / len(batch)
        self.log.screened(prompt, generation, screen.score, before)
        if screen.score <= before:
            return None
        return self._score(prompt, generation)

    # empty or too long falls back to the input
    def _operate(self, operator, fallback):
        text = self._safely(operator, fallback)
        if not text or len(text) > self.config.max_prompt_chars:
            return fallback
        return text

    # best mu unique. shuffle first so ties are random
    def _survivors(self, pool):
        unique = list({ind.prompt: ind for ind in pool}.values())
        self._rng.shuffle(unique)
        unique.sort(key=lambda ind: ind.evaluation.score, reverse=True)
        return unique[: self.config.mu]
