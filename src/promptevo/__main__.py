"""CLI for prompt evolution. From src/:

    python -m promptevo Python/0 Python/1 --held-out Python/2 Python/3

1. Score the seeds on the train items, keep the best mu
2. Evolve (search/evolution.py), or run GEPA's loop with --search gepa
   (unused/gepa.py)
3. With --validation, score the final population and the seeds on the
   validation items and pick the best. GEPA uses the validation items
   during the search instead
4. Score the best prompt, the validation pick, the best seed, no prompt and
   every seed on the held-out items (the test set)
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from promptevo.datasets.human_eval.dataset import FORMATS, HumanEval
from promptevo.llm.models import mutator, solver
from promptevo.paths import RESULTS
from promptevo.scoring.evaluation import Evaluator, Signal
from promptevo.scoring.experiment import (
    pick_by_validation,
    score_test_set,
    test_prompts,
)
from promptevo.scoring.individual import Individual
from promptevo.search.base import Search
from promptevo.search.evolution import Evolution, EvolutionConfig
from promptevo.search.feedback import Feedback
from promptevo.search.mutations import MUTATIONS, MixedOperators
from promptevo.search.runlog import RunLog
from promptevo.search.selection import UniformSelector
from promptevo.search.templates import SEED_PROMPTS
from promptevo.unused.gepa import GEPA
from promptevo.unused.selection import (
    DownsampledLexicaseSelector,
    GreedySelector,
    LexicaseSelector,
    ParetoSelector,
    RouletteSelector,
)
from promptevo.unused.terminal_bench import TerminalBench

# {--dataset name: Dataset class}. add new ones here
DATASETS = {
    "humaneval": HumanEval,
    "terminal-bench": TerminalBench,  # not in the paper
}

# {--selection name: Selector class}. the study only used uniform
SELECTORS = {
    "uniform": UniformSelector,
    "lexicase": LexicaseSelector,
    "lexicase-down": DownsampledLexicaseSelector,
    "pareto": ParetoSelector,
    "roulette": RouletteSelector,
    "greedy": GreedySelector,
}


# what a search ended with, for the test set and the drift check
class _Searched(NamedTuple):
    search:     Search
    best:       Individual
    picked:     str | None               # validation pick, ea only
    validation: dict[str, float] | None  # {prompt: validation score}


def _parser():
    parser = argparse.ArgumentParser(prog="promptevo")
    parser.add_argument("items", nargs="+", help="train items")
    _add_data_args(parser.add_argument_group("data"))
    _add_model_args(parser.add_argument_group("models"))
    _add_search_args(parser.add_argument_group("search"))
    out = parser.add_argument_group("output")
    out.add_argument(
        "--log", type=Path, help="default results/evolution-<time>.jsonl"
    )
    out.add_argument(
        "--drift-check", action="store_true",
        help="re-score the best seed with nothing cached at the end",
    )
    return parser


def _add_data_args(group):
    group.add_argument("--validation", nargs="*", default=[])
    group.add_argument(
        "--held-out", nargs="*", default=[], help="the test set"
    )
    group.add_argument(
        "--dataset", choices=list(DATASETS), default="humaneval"
    )
    group.add_argument(
        "--data-dir", type=Path, help="the dataset picks if not set"
    )
    group.add_argument(
        "--reply-format", choices=list(FORMATS), default="code",
        help="what the solver is asked to reply with, humaneval only",
    )
    group.add_argument(
        "--sandbox-cache", type=Path,
        help="sqlite file of sandbox results runs share, humaneval only",
    )


def _add_model_args(group):
    group.add_argument(
        "--solver", default="qwen2.5-coder:7b",
        help="ollama model that fixes the code, or with --vllm-url the "
        "server's --served-model-name",
    )
    group.add_argument(
        "--mutator", default="qwen2.5-coder:7b",
        help="ollama model that edits prompts",
    )
    group.add_argument(
        "--vllm-url", help="run the solver on a vllm server instead"
    )
    group.add_argument(
        "--cache", type=Path,
        help="sqlite file of solver replies runs share, vllm only",
    )
    group.add_argument(
        "--max-tokens", type=int, default=1024,
        help="longest solver reply, vllm only. formats that reason need more",
    )


def _add_search_args(group):
    group.add_argument(
        "--search", choices=["ea", "gepa"], default="ea",
        help="search/evolution.py or unused/gepa.py",
    )
    group.add_argument("--mu", type=int, default=2)
    group.add_argument("--lambda", dest="lambda_", type=int, default=2)
    group.add_argument("--generations", type=int, default=2)
    group.add_argument("--mutation-rate", type=float, default=0.5)
    group.add_argument("--crossover-rate", type=float, default=1.0)
    group.add_argument(
        "--screen", type=int, default=0,
        help="score a child on this many train items first, fully only if "
        "it beats its parent there",
    )
    group.add_argument(
        "--budget", type=int, default=0,
        help="solver calls before stopping, 0 means no limit",
    )
    group.add_argument(
        "--selection", choices=list(SELECTORS),
        help="default uniform for ea, pareto for gepa",
    )
    group.add_argument(
        "--signal", type=Signal, choices=list(Signal),
        default=Signal.PARTIAL, help="fitness the search selects on",
    )
    group.add_argument(
        "--feedback", type=Feedback, choices=list(Feedback),
        default=Feedback.NONE, help="what the rewrite mutation sees",
    )
    group.add_argument(
        "--mutations", nargs="+", choices=list(MUTATIONS),
        default=["rewrite"], help="one is picked at random for every child",
    )
    group.add_argument(
        "--hide-tests", action="store_true",
        help="hide the failing asserts from reflective mutation, they hold "
        "the expected values",
    )
    group.add_argument("--comma", action="store_true", help="children only")
    group.add_argument("--seed", type=int, default=0)


# the dataset on items, with the humaneval-only options that are set.
# cached=False leaves out the sandbox cache
def _dataset(args, items, cached=True):
    options = {}
    if args.sandbox_cache and cached:
        options["sandbox_cache"] = args.sandbox_cache
    if args.reply_format != "code":
        options["reply_format"] = args.reply_format
    if options and args.dataset != "humaneval":
        raise SystemExit(
            "--sandbox-cache and --reply-format only work with humaneval"
        )
    return DATASETS[args.dataset](items, args.data_dir, **options)


def _operators(args, dataset, log):
    return MixedOperators(
        mutator(args.mutator),
        [MUTATIONS[name]() for name in args.mutations],
        dataset.description,
        args.feedback,
        args.seed,
        log,
        show_tests=not args.hide_tests,
    )


def _config(args):
    return EvolutionConfig(
        mu=args.mu,
        lambda_=args.lambda_,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        plus=not args.comma,
        seed=args.seed,
        crossover_rate=args.crossover_rate,
        screen=args.screen,
        budget=args.budget,
    )


def _run_ea(args, train, model, operators, log):
    selector = SELECTORS[args.selection or "uniform"]()
    search = Evolution(_config(args), train, operators, selector, log)
    population = search.run(SEED_PROMPTS)
    best = population[0]
    print(f"\nbest prompt (train score {best.evaluation.score:.3f}):")
    print(best.prompt)
    picked = validation = None
    if args.validation:
        picked, validation = _validation(
            args, model, search, population, log
        )
    return _Searched(search, best, picked, validation)


def _run_gepa(args, train, model, operators, log):
    if not args.validation or not args.budget:
        raise SystemExit("gepa needs --validation and --budget")
    pareto = Evaluator(_dataset(args, args.validation), model, Signal.PARTIAL)
    search = GEPA(
        train,
        pareto,
        args.items,
        operators,
        args.budget,
        seed=args.seed,
        log=log,
        selector=SELECTORS[args.selection or "pareto"](),
    )
    best = search.run(SEED_PROMPTS)[0]
    print(f"\nbest prompt (validation score {best.evaluation.score:.3f}):")
    print(best.prompt)
    validation = {s.prompt: s.evaluation.score for s in search.seeds}
    return _Searched(search, best, None, validation)


# best of the final population and the seeds on the validation items. the
# best train score is often just the best fit to the train items
def _validation(args, model, evolution, population, log):
    dataset = _dataset(args, args.validation)
    validator = Evaluator(dataset, model, Signal.PARTIAL)
    candidates = [ind.prompt for ind in population + evolution.seeds]
    picked, scores = pick_by_validation(validator, candidates)
    log.validation(scores)
    print(f"\nvalidation pick (validation score {scores[picked]:.3f}):")
    print(picked)
    return picked, scores


def _held_out(args, model, searched, log):
    # always partial here, same metric no matter which signal trained it
    tester = Evaluator(_dataset(args, args.held_out), model, Signal.PARTIAL)
    prompts = test_prompts(
        searched.best,
        searched.search.seeds,
        searched.picked,
        searched.validation,
    )
    tested = score_test_set(tester, prompts)
    log.test(prompts, tested)
    print("\ntest set (partial credit):")
    for name, evaluation in tested.items():
        print(f"  {name:<10} {evaluation.score:.3f}")


# re-scores the best seed with nothing cached, on the items it was scored on
# at the start. a different score means the solver or sandbox drifted, so
# it gets its own solver and dataset without --cache or --sandbox-cache
def _drift_check(args, search, log):
    gepa = args.search == "gepa"
    items = args.validation if gepa else args.items
    signal = Signal.PARTIAL if gepa else args.signal
    seed = max(search.seeds, key=lambda ind: ind.evaluation.score)
    dataset = _dataset(args, items, cached=False)
    model = solver(args.solver, args.vllm_url, max_tokens=args.max_tokens)
    after = Evaluator(dataset, model, signal).evaluate(seed.prompt).score
    log.drift(seed.prompt, seed.evaluation.score, after)
    print(f"\ndrift check: best seed {seed.evaluation.score:.3f}, "
          f"now {after:.3f}")


def main():
    args = _parser().parse_args()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = args.log or RESULTS / f"evolution-{stamp}.jsonl"
    model = solver(args.solver, args.vllm_url, args.cache, args.max_tokens)
    dataset = _dataset(args, args.items)
    log = RunLog(log_path)
    operators = _operators(args, dataset, log)
    train = Evaluator(dataset, model, args.signal)
    run = _run_gepa if args.search == "gepa" else _run_ea
    searched = run(args, train, model, operators, log)
    if args.held_out:
        _held_out(args, model, searched, log)
    if args.drift_check:
        _drift_check(args, searched.search, log)
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
