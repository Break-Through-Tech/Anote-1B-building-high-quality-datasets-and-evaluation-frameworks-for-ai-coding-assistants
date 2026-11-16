"""CLI for prompt evolution. From src/:

    python -m promptevo task-a--bug01 task-b--bug03 --held-out task-c--bug02 \\
        --data-dir ../data/processed/bug-injected/tasks

1. Score the seeds on the train items, keep the best mu
2. Evolve (evolution.py), or run GEPA's loop with --search gepa (gepa.py)
3. With --validation, score the final population and the seeds on the
   validation items and pick the best. GEPA uses the validation items
   during the search instead
4. Score the best prompt, the validation pick, the best seed, no prompt and
   every seed on the held-out items (the test set)
"""

import argparse
from datetime import datetime
from pathlib import Path

from promptevo.datasets import DATASETS
from promptevo.datasets.human_eval.dataset import FORMATS
from promptevo.evaluation import Evaluator, Signal
from promptevo.evolution import Evolution, EvolutionConfig, RunLog
from promptevo.gepa import GEPA
from promptevo.experiment import (
    SEED_PROMPTS,
    pick_by_validation,
    score_test_set,
    test_prompts,
)
from promptevo.models import mutator, solver
from promptevo.mutations import MUTATIONS, MixedOperators
from promptevo.operators import Feedback
from promptevo.selection import (
    DownsampledLexicaseSelector,
    GreedySelector,
    LexicaseSelector,
    ParetoSelector,
    RouletteSelector,
    UniformSelector,
)

SELECTORS = {
    "uniform": UniformSelector,
    "lexicase": LexicaseSelector,
    "lexicase-down": DownsampledLexicaseSelector,
    "pareto": ParetoSelector,
    "roulette": RouletteSelector,
    "greedy": GreedySelector,
}

_RESULTS_DIR = Path(__file__).parents[2] / "results"


def _parse_args():
    parser = argparse.ArgumentParser(prog="promptevo")

    # train items
    parser.add_argument("items", nargs="+")
    parser.add_argument("--validation", nargs="*", default=[])
    parser.add_argument("--held-out", nargs="*", default=[])
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS),
        default="terminal-bench",
    )
    # dataset picks if not set
    parser.add_argument("--data-dir", type=Path)

    # ollama models. solver fixes code, mutator edits prompts
    parser.add_argument("--solver", default="qwen2.5-coder:7b")
    parser.add_argument("--mutator", default="qwen2.5-coder:7b")
    # run the solver on a vllm server instead, e.g. http://localhost:8000.
    # --solver is then the server's --served-model-name
    parser.add_argument("--vllm-url")
    # sqlite file of solver replies to share between runs, vllm only
    parser.add_argument("--cache", type=Path)
    # sqlite file of sandbox results to share between runs, humaneval only
    parser.add_argument("--sandbox-cache", type=Path)
    # what the solver is asked to reply with, humaneval only. see FORMATS
    parser.add_argument(
        "--reply-format", choices=list(FORMATS), default="code"
    )
    # longest solver reply, vllm only. formats that reason need more
    parser.add_argument("--max-tokens", type=int, default=1024)

    parser.add_argument("--mu", type=int, default=2)
    parser.add_argument("--lambda", dest="lambda_", type=int, default=2)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--mutation-rate", type=float, default=0.5)
    parser.add_argument("--crossover-rate", type=float, default=1.0)
    parser.add_argument("--screen", type=int, default=0)
    # ea is evolution.py, gepa is gepa.py. budget is in solver calls
    parser.add_argument("--search", choices=["ea", "gepa"], default="ea")
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--drift-check", action="store_true")
    # hide failing asserts from reflective mutation, they hold the
    # expected values
    parser.add_argument("--hide-tests", action="store_true")
    parser.add_argument("--comma", action="store_true", help="children only")
    parser.add_argument("--seed", type=int, default=0)

    # things to change for experiments!
    parser.add_argument(
        "--selection",
        choices=list(SELECTORS),
        # default uniform for ea, pareto for gepa
    )
    parser.add_argument(
        "--signal",
        type=Signal,
        choices=list(Signal),
        default=Signal.PARTIAL,
    )
    parser.add_argument(
        "--feedback",
        type=Feedback,
        choices=list(Feedback),
        default=Feedback.NONE,
    )
    # one is picked at random for every mutation
    parser.add_argument(
        "--mutations",
        nargs="+",
        choices=list(MUTATIONS),
        default=["rewrite"],
    )

    # default results/evolution-<time>.jsonl
    parser.add_argument("--log", type=Path)
    return parser.parse_args()


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


def _held_out(args, model, evolution, best, picked, validation, log):
    # always partial here, same metric no matter which signal trained it
    dataset = _dataset(args, args.held_out)
    tester = Evaluator(dataset, model, Signal.PARTIAL)
    prompts = test_prompts(best, evolution.seeds, picked, validation)
    scores = score_test_set(tester, prompts)
    # already scored, this only reads the evaluator's cache
    results = {name: tester.evaluate(p).results for name, p in prompts.items()}
    log.test(prompts, scores, results)
    print("\ntest set (partial credit):")
    for name, score in scores.items():
        print(f"  {name:<10} {score:.3f}")


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


if __name__ == "__main__":
    args = _parse_args()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = args.log or _RESULTS_DIR / f"evolution-{stamp}.jsonl"
    config = EvolutionConfig(
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
    solver_model = solver(
        args.solver, args.vllm_url, args.cache, args.max_tokens
    )
    dataset = _dataset(args, args.items)
    log = RunLog(log_path)
    operators = MixedOperators(
        mutator(args.mutator),
        [MUTATIONS[name]() for name in args.mutations],
        dataset.description,
        args.feedback,
        args.seed,
        log,
        show_tests=not args.hide_tests,
    )
    train = Evaluator(dataset, solver_model, args.signal)
    picked = validation = None
    if args.search == "gepa":
        if not args.validation or not args.budget:
            raise SystemExit("gepa needs --validation and --budget")
        search = GEPA(
            train,
            Evaluator(
                _dataset(args, args.validation),
                solver_model,
                Signal.PARTIAL,
            ),
            args.items,
            operators,
            args.budget,
            seed=args.seed,
            log=log,
            selector=SELECTORS[args.selection or "pareto"](),
        )
        population = search.run(SEED_PROMPTS)
        best = population[0]
        validation = {s.prompt: s.evaluation.score for s in search.seeds}
        print(f"\nbest prompt (validation score {best.evaluation.score:.3f}):")
        print(best.prompt)
    else:
        search = Evolution(
            config,
            train,
            operators,
            SELECTORS[args.selection or "uniform"](),
            log,
        )
        population = search.run(SEED_PROMPTS)
        best = population[0]
        print(f"\nbest prompt (train score {best.evaluation.score:.3f}):")
        print(best.prompt)
        if args.validation:
            picked, validation = _validation(
                args, solver_model, search, population, log
            )
    if args.held_out:
        _held_out(
            args, solver_model, search, best, picked, validation, log
        )
    if args.drift_check:
        _drift_check(args, search, log)
    print(f"log: {log_path}")
