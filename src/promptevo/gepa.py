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
import sys

from promptevo.evaluation import Evaluator
from promptevo.evolution import RunLog, mutator_calls
from promptevo.operators import Operators
from promptevo.selection import ParetoSelector, Selector
from promptevo.types import Evaluation, Individual


class GEPA:
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
        self.feedback  = feedback
        self.pareto    = pareto
        self.items     = feedback_items
        self.operators = operators
        self.budget    = budget
        self.minibatch = minibatch
        self.log       = log
        # separate streams, so runs that only differ in selection still see
        # the same minibatches
        self._pick_rng  = random.Random(seed)
        self._batch_rng = random.Random(seed + 1)
        # gepa's own is pareto. anything else is our change to it
        self._selector = selector or ParetoSelector()

    @property
    def rollouts(self) -> int:
        """solver calls so far, train and validation."""
        return self.feedback.rollouts + self.pareto.rollouts

    def run(self, seeds: list[str]) -> list[Individual]:
        """runs until the budget is spent. returns the pool, best first."""
        self.seeds = [self._add(prompt, 0) for prompt in seeds]
        pool = list(self.seeds)
        self.operators.observe(_ranked(pool))
        tried = set(seeds)
        step = 0
        # every step costs a minibatch unless the mutation came back empty,
        # the cap stops a mutator that keeps failing from looping forever
        while self.rollouts < self.budget and step < 10 * self.budget:
            step += 1
            parent = self._selector.pick(pool, self._pick_rng)
            batch = self._batch_rng.sample(self.items, self.minibatch)
            before = self._minibatch(parent.prompt, batch)
            if before is None:
                continue
            child = self._mutate(parent.prompt, before)
            if not child or child in tried:
                continue
            tried.add(child)
            after = self._minibatch(child, batch)
            if after is None:
                continue
            after = after.score
            if self.log:
                self.log.screened(child, step, after, before.score)
            if after > before.score:
                pool.append(self._add(child, step))
                self.operators.observe(_ranked(pool))
                if self.log:
                    self.log.generation(
                        step,
                        _ranked(pool),
                        self.rollouts,
                        mutator_calls(self.operators),
                    )
        return _ranked(pool)

    # scores prompt on the validation items. a crash scores 0
    def _add(self, prompt, step):
        try:
            evaluation = self.pareto.evaluate(prompt)
        except Exception as e:
            why = f"evaluation failed: {e!r}"
            print(why, file=sys.stderr)
            evaluation = Evaluation(0.0, {}, why, crashed=True)
        ind = Individual(prompt, evaluation, step)
        if self.log:
            self.log.scored(ind)
        return ind

    # a crash skips the step instead of ending a long run
    def _minibatch(self, prompt, batch):
        try:
            return self.feedback.evaluate(prompt, batch)
        except Exception as e:
            print(f"evaluation failed: {e!r}", file=sys.stderr)
            return None

    def _mutate(self, prompt, evaluation):
        try:
            return self.operators.mutate(prompt, evaluation).strip()
        except Exception as e:
            print(f"operator failed: {e!r}", file=sys.stderr)
            return ""


# best validation score first, ties go to the older prompt
def _ranked(pool):
    return sorted(pool, key=lambda ind: -ind.evaluation.score)
