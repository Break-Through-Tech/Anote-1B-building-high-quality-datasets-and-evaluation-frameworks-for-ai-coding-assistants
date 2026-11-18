"""GEPA's search loop (Agrawal et al., alg. 1). https://arxiv.org/abs/2507.19457

Every step:
1. Pick a prompt from the pool with pareto selection over the validation
   items (the paper's D_pareto)
2. Run it on a minibatch of 3 train items (D_feedback) and mutate it with
   what happened there, normally the reflective mutation
3. Run the child on the same minibatch. Only if it scores higher does it get
   scored on the validation items and join the pool

Stops when the solver has been called budget times, on train and validation
together, and returns the prompt with the best validation score.
"""

import random

from promptevo.scoring.evaluation import Evaluator
from promptevo.scoring.individual import Individual
from promptevo.search.base import Search
from promptevo.search.operators import Operators
from promptevo.search.runlog import RunLog
from promptevo.search.selection import Selector
from promptevo.unused.selection import ParetoSelector


class GEPA(Search):
    """the gepa loop. feedback scores minibatches, pareto scores the pool."""

    def __init__(
        self,
        feedback: Evaluator,
        pareto: Evaluator,
        feedback_items: list[str],
        operators: Operators,
        budget: int,
        minibatch: int = 3,
        seed: int = 0,
        log: RunLog | None = None,
        selector: Selector | None = None,
    ):
        super().__init__(operators, log)
        self.feedback  = feedback
        self.pareto    = pareto
        self.items     = feedback_items
        self.budget    = budget
        self.minibatch = minibatch
        # separate streams, so runs that only differ in selection still see
        # the same minibatches
        self._pick_rng  = random.Random(seed)
        self._batch_rng = random.Random(seed + 1)
        # gepa's own is pareto. anything else is our change to it
        self._selector = selector or ParetoSelector()

    @property
    def rollouts(self):
        """solver calls so far, train and validation."""
        return self.feedback.rollouts + self.pareto.rollouts

    def run(self, seeds: list[str]) -> list[Individual]:
        """runs until the budget is spent. returns the pool, best first."""
        self.seeds = [self._scored(self.pareto, p, 0) for p in seeds]
        pool = list(self.seeds)
        self.operators.observe(_ranked(pool))
        tried = set(seeds)
        step = 0
        # every step costs a minibatch unless the mutation came back empty,
        # the cap stops a mutator that keeps failing from looping forever
        while self.rollouts < self.budget and step < 10 * self.budget:
            step += 1
            child = self._step(pool, tried, step)
            if child is not None:
                pool.append(child)
                self._new_population(step, _ranked(pool))
        return _ranked(pool)

    # one parent, one child. returns the child if it beat its parent on the
    # minibatch, scored on the validation items
    def _step(self, pool, tried, step):
        parent = self._selector.pick(pool, self._pick_rng)
        batch = self._batch_rng.sample(self.items, self.minibatch)
        before = self.feedback.try_evaluate(parent.prompt, batch)
        if before.crashed:
            return None
        child = self._safely(
            lambda: self.operators.mutate(parent.prompt, before), ""
        )
        if not child or child in tried:
            return None
        tried.add(child)
        after = self.feedback.try_evaluate(child, batch)
        if after.crashed:
            return None
        self.log.screened(child, step, after.score, before.score)
        if after.score <= before.score:
            return None
        return self._scored(self.pareto, child, step)


# best validation score first, ties go to the older prompt
def _ranked(pool):
    return sorted(pool, key=lambda ind: -ind.evaluation.score)
