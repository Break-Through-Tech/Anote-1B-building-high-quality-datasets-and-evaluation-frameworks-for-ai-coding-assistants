"""(mu+lambda) prompt evolution. The evaluator, operators and selector get
passed in.

Per generation:
1. Pick 2 parents (Selector), or 1 when the child skips crossover
2. Crossover, maybe mutate. Without crossover, always mutate
3. Score new children. With screening, only the ones that beat their parent
   on a few train items first
4. Keep the best mu of parents + children (just children with comma).
   Ties are random

Logs every scored prompt and generation to JSONL. https://jsonlines.org/
"""

import json
import random
import sys
from pathlib import Path
from typing import NamedTuple

from promptevo.datasets.base import ItemResult
from promptevo.evaluation import Evaluator, binary, naive, partial
from promptevo.operators import Operators
from promptevo.selection import Selector, UniformSelector
from promptevo.types import Evaluation, Individual


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


class RunLog:
    """jsonl log, appended right away so a crash doesn't lose it."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def scored(self, ind: Individual):
        self._write({
            "event": "scored",
            "generation": ind.generation,
            "prompt": ind.prompt,
            "score": ind.evaluation.score,
            "items": ind.evaluation.items,
            "cases": ind.evaluation.cases,
            "feedback": ind.evaluation.feedback,
            "crashed": ind.evaluation.crashed,
        })

    def generation(
        self,
        generation: int,
        population: list[Individual],
        rollouts: int | None = None,
        mutator_calls: int | None = None,
    ):
        self._write({
            "event": "generation",
            "generation": generation,
            "survivors": [
                {"prompt": p.prompt, "score": p.evaluation.score}
                for p in population
            ],
            "rollouts": rollouts,
            "mutator_calls": mutator_calls,
        })

    def screened(
        self, prompt: str, generation: int, score: float, parent: float
    ):
        self._write({
            "event": "screened",
            "generation": generation,
            "prompt": prompt,
            "score": score,
            "parent_score": parent,
        })

    def mutation(self, kind: str, parent: str, child: str):
        self._write({
            "event": "mutation",
            "kind": kind,
            "parent": parent,
            "child": child,
        })

    def validation(self, scores: dict[str, float]):
        self._write({
            "event": "validation",
            "results": [
                {"prompt": prompt, "score": score}
                for prompt, score in scores.items()
            ],
        })

    def drift(self, prompt: str, before: float, after: float):
        self._write({
            "event": "drift",
            "prompt": prompt,
            "before": before,
            "after": after,
        })

    def test(
        self,
        prompts: dict[str, str],
        scores: dict[str, float],
        results: dict[str, dict[str, ItemResult]] | None = None,
    ):
        """the test set scores. with results ({name: {item: ItemResult}})
        every item's scores and reply go in too."""
        record = {
            name: {"prompt": prompts[name], "score": scores[name]}
            for name in prompts
        }
        for name, items in (results or {}).items():
            record[name]["items"] = {
                item: {
                    "partial": partial(result),
                    "binary": binary(result),
                    "naive": naive(result),
                    "output": result.output,
                }
                for item, result in items.items()
            }
        self._write({"event": "test", "results": record})

    def _write(self, record):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


class Evolution:
    """the ea loop"""

    def __init__(
        self,
        config: EvolutionConfig,
        evaluator: Evaluator,
        operators: Operators,
        selector: Selector | None = None,
        log: RunLog | None = None,
    ):
        self.config = config
        self.evaluator = evaluator
        self.operators = operators
        self.selector = selector or UniformSelector()
        self.log = log

    def run(self, seeds: list[str]) -> list[Individual]:
        """runs the ea. returns final population, sorted by best"""
        self._check(len(seeds))
        self._rng = random.Random(self.config.seed)
        self._scored = {}  # {prompt: Individual}

        # kept so the test set can compare against the best seed
        self.seeds = [self._score(p, 0) for p in seeds]
        if all(ind.evaluation.crashed for ind in self.seeds):
            raise RuntimeError(
                "every seed evaluation crashed :(, check docker/ollama!"
            )
        population = self._survivors(self.seeds)
        self._log_generation(0, population)
        for generation in range(1, self.config.generations + 1):
            if self._spent():
                break
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
            self._log_generation(generation, population)
        return population

    # checked before every child, so a run goes over by at most one
    # child's full score
    def _spent(self):
        budget = self.config.budget
        return bool(budget) and (
            getattr(self.evaluator, "rollouts", 0) >= budget
        )

    def _check(self, n_seeds):
        config = self.config
        if config.mu < 2:
            raise ValueError("crossover needs mu >= 2")
        if n_seeds < config.mu:
            raise ValueError(f"need at least mu={config.mu} seed prompts")
        if not config.plus and config.lambda_ < config.mu:
            raise ValueError("comma selection needs lambda_ >= mu")

    def _score(self, prompt, generation):
        if prompt not in self._scored:
            ind = Individual(prompt, self._safe_eval(prompt), generation)
            self._scored[prompt] = ind
            if self.log:
                self.log.scored(ind)
        return self._scored[prompt]

    # a crash scores 0 instead of killing the run
    def _safe_eval(self, prompt):
        try:
            return self.evaluator.evaluate(prompt)
        except Exception as e:
            why = f"evaluation failed: {e!r}"
            print(why, file=sys.stderr)
            return Evaluation(0.0, {}, why, crashed=True)

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
        if not size or prompt in self._scored or not items:
            return self._score(prompt, generation)
        batch = self._rng.sample(sorted(items), min(size, len(items)))
        try:
            score = self.evaluator.evaluate(prompt, batch).score
        except Exception as e:
            print(f"evaluation failed: {e!r}", file=sys.stderr)
            return None
        before = sum(items[i] for i in batch) / len(batch)
        if self.log:
            self.log.screened(prompt, generation, score, before)
        if score <= before:
            return None
        return self._score(prompt, generation)

    def _operate(self, op, fallback):
        try:
            text = op().strip()
        except Exception as e:
            print(f"operator failed: {e!r}", file=sys.stderr)
            return fallback
        if not text or len(text) > self.config.max_prompt_chars:
            return fallback
        return text

    # best mu unique. shuffle first so ties are random
    def _survivors(self, pool):
        unique = list({ind.prompt: ind for ind in pool}.values())
        self._rng.shuffle(unique)
        unique.sort(key=lambda ind: ind.evaluation.score, reverse=True)
        return unique[: self.config.mu]

    def _log_generation(self, generation, population):
        self.operators.observe(population)
        if self.log:
            rollouts = getattr(self.evaluator, "rollouts", None)
            calls = mutator_calls(self.operators)
            self.log.generation(generation, population, rollouts, calls)


def mutator_calls(operators: Operators) -> int | None:
    """chat calls the mutator model has made, if the operators have one."""
    model = getattr(operators, "model", None)
    return getattr(model, "calls", None)
