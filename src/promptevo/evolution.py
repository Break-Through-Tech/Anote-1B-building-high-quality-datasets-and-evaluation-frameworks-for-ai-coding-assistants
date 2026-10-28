"""(mu+lambda) prompt evolution. The evaluator, operators and selector get
passed in.

Per generation:
1. Pick 2 parents (Selector)
2. Crossover, maybe mutate
3. Score new children
4. Keep the best mu of parents + children (just children with comma).
   Ties are random

Logs every scored prompt and generation to JSONL. https://jsonlines.org/
"""

import json
import random
import sys
from pathlib import Path
from typing import NamedTuple

from promptevo.evaluation import Evaluator
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
            "cases": ind.evaluation.cases,
            "feedback": ind.evaluation.feedback,
            "crashed": ind.evaluation.crashed,
        })

    def generation(self, generation: int, population: list[Individual]):
        self._write({
            "event": "generation",
            "generation": generation,
            "survivors": [
                {"prompt": p.prompt, "score": p.evaluation.score}
                for p in population
            ],
        })

    def test(self, prompts: dict[str, str], scores: dict[str, float]):
        self._write({
            "event": "test",
            "results": {
                name: {"prompt": prompts[name], "score": scores[name]}
                for name in prompts
            },
        })

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
            children = [
                self._score(self._child(population), generation)
                for _ in range(self.config.lambda_)
            ]
            pool = population + children if self.config.plus else children
            population = self._survivors(pool)
            self._log_generation(generation, population)
        return population

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

    # crossover + maybe mutate. bad output falls back to the input
    def _child(self, population):
        ops = self.operators
        a, b = self.selector.parents(population, self._rng)
        child = self._operate(
            lambda: ops.crossover(a.prompt, b.prompt), a.prompt
        )
        if self._rng.random() < self.config.mutation_rate:
            # child has no score yet, use parent a's
            child = self._operate(
                lambda: ops.mutate(child, a.evaluation), child
            )
        return child

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
        if self.log:
            self.log.generation(generation, population)
