"""The EA and GEPA loops with a fake solver and mutator. Their logs pin the
random draws, so a refactor that reorders them shows up here."""

import hashlib

import pytest

from conftest import RecordingModel, check_golden, evaluation
from promptevo.__main__ import SELECTORS
from promptevo.scoring.evaluation import Evaluator
from promptevo.search.evolution import Evolution, EvolutionConfig
from promptevo.search.mutations import MUTATIONS, MixedOperators
from promptevo.search.operators import Operators
from promptevo.search.runlog import LoggedRun, RunLog
from promptevo.unused.gepa import GEPA

ITEMS = ["a", "b", "c", "d", "e"]


# 0, 0.25, ..., 1 from a hash, so every prompt has fixed item scores
def _fake_score(prompt, item):
    digest = hashlib.sha256(f"{prompt}/{item}".encode()).digest()
    return digest[0] % 5 / 4


# an Evaluator whose scores come from a hash instead of a dataset
class FakeEvaluator(Evaluator):
    def __init__(self, items=ITEMS):
        self.items = items
        self.rollouts = 0

    def evaluate(self, prompt, items=None):
        names = items or self.items
        self.rollouts += len(names)
        return evaluation({item: _fake_score(prompt, item) for item in names})


class FakeOperators(Operators):
    def __init__(self):
        self.made = 0
        self.seen = []

    def mutate(self, prompt, evaluation):
        self.made += 1
        return f"{prompt[:30]} m{self.made}"

    def crossover(self, a, b):
        self.made += 1
        return f"{a[:15]}|{b[:15]} x{self.made}"

    def observe(self, population):
        self.seen.append([ind.prompt for ind in population])


SEEDS = ["Seed A", "Seed B", "Seed C", "Seed D"]


def _events(path):
    return LoggedRun(path).events


def _evolve(tmp_path, selector, operators=None, **config):
    log = RunLog(tmp_path / "run.jsonl")
    settings = dict(mu=3, lambda_=4, generations=4, seed=7) | config
    evolution = Evolution(
        EvolutionConfig(**settings),
        FakeEvaluator(),
        operators or FakeOperators(),
        SELECTORS[selector](),
        log,
    )
    population = evolution.run(SEEDS)
    return {
        "final": [ind.prompt for ind in population],
        "events": _events(log.path),
    }


@pytest.mark.parametrize("selector", list(SELECTORS))
def test_evolution_with_every_selector(tmp_path, selector):
    check_golden(
        f"evolution-{selector}",
        _evolve(tmp_path, selector, crossover_rate=0.5),
    )


def test_evolution_variants(tmp_path):
    variants = {
        "comma": dict(plus=False, lambda_=5),
        "screened": dict(screen=2, crossover_rate=0),
        "budget": dict(budget=60),
        "always crossover": dict(mutation_rate=1.0),
    }
    out = {}
    for name, config in variants.items():
        folder = tmp_path / name
        out[name] = _evolve(folder, "uniform", **config)
    check_golden("evolution-variants", out)


def test_evolution_with_mixed_operators(tmp_path):
    model = RecordingModel()
    names = ["rewrite", "first-order", "eda", "zero-order-hyper"]
    log = RunLog(tmp_path / "mixed.jsonl")
    operators = MixedOperators(
        model, [MUTATIONS[n]() for n in names], "Fix it.", seed=4, log=log
    )
    evolution = Evolution(
        EvolutionConfig(mu=3, lambda_=3, generations=3, seed=4,
                        crossover_rate=0),
        FakeEvaluator(),
        operators,
        SELECTORS["uniform"](),
        log,
    )
    evolution.run(SEEDS)
    check_golden("evolution-mixed", {
        "events": _events(log.path),
        "sent": model.sent,
    })


def test_gepa(tmp_path):
    log = RunLog(tmp_path / "gepa.jsonl")
    gepa = GEPA(
        FakeEvaluator(ITEMS),
        FakeEvaluator(["v1", "v2", "v3"]),
        ITEMS,
        FakeOperators(),
        budget=80,
        seed=2,
        log=log,
    )
    pool = gepa.run(SEEDS)
    check_golden("gepa", {
        "pool": [ind.prompt for ind in pool],
        "rollouts": gepa.rollouts,
        "events": _events(log.path),
    })


def test_test_event_format(tmp_path):
    log = RunLog(tmp_path / "test.jsonl")
    log.test({"best": "Seed A"}, {"best": evaluation({"a": 1.0, "b": 0.0})})
    check_golden("test-event", _events(log.path))


# a seed whose evaluation throws scores 0 and the run carries on, so do
# children whose screening or full score throws
class CrashingEvaluator(FakeEvaluator):
    def evaluate(self, prompt, items=None):
        if "B" in prompt or prompt.endswith("m2"):
            raise RuntimeError(f"sandbox died on {prompt}")
        return super().evaluate(prompt, items)


def test_crashes_score_zero(tmp_path):
    log = RunLog(tmp_path / "crash.jsonl")
    evolution = Evolution(
        EvolutionConfig(mu=3, lambda_=3, generations=2, seed=1,
                        crossover_rate=0, screen=2),
        CrashingEvaluator(),
        FakeOperators(),
        SELECTORS["uniform"](),
        log,
    )
    population = evolution.run(SEEDS)
    check_golden("evolution-crashes", {
        "final": [ind.prompt for ind in population],
        "events": _events(log.path),
    })
