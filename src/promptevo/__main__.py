"""CLI for prompt evolution. From src/:

    python -m promptevo task-a--bug01 task-b--bug03 --held-out task-c--bug02 \\
        --data-dir ../data/processed/bug-injected/tasks

1. Score the seeds on the train items, keep the best mu
2. Evolve (evolution.py)
3. Score the best prompt, the best seed, no prompt and every seed on the
   held-out items (the test set)
"""

import argparse
from datetime import datetime
from pathlib import Path

from promptevo.datasets import DATASETS
from promptevo.evaluation import Evaluator, Signal
from promptevo.evolution import Evolution, EvolutionConfig, RunLog
from promptevo.experiment import SEED_PROMPTS, score_test_set, test_prompts
from promptevo.models import mutator, solver
from promptevo.operators import Feedback, LLMOperators
from promptevo.selection import LexicaseSelector, UniformSelector

SELECTORS = {
    "uniform": UniformSelector,
    "lexicase": LexicaseSelector,
}

_RESULTS_DIR = Path(__file__).parents[2] / "results"


def _parse_args():
    parser = argparse.ArgumentParser(prog="promptevo")

    # train items
    parser.add_argument("items", nargs="+")
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

    parser.add_argument("--mu", type=int, default=2)
    parser.add_argument("--lambda", dest="lambda_", type=int, default=2)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--mutation-rate", type=float, default=0.5)
    parser.add_argument("--comma", action="store_true", help="children only")
    parser.add_argument("--seed", type=int, default=0)

    # things to change for experiments!
    parser.add_argument(
        "--selection",
        choices=list(SELECTORS),
        default="uniform",
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

    # default results/evolution-<time>.jsonl
    parser.add_argument("--log", type=Path)
    return parser.parse_args()


def _held_out(args, model, evolution, best, log):
    # always partial here, same metric no matter which signal trained it
    dataset = DATASETS[args.dataset](args.held_out, args.data_dir)
    tester = Evaluator(dataset, model, Signal.PARTIAL)
    prompts = test_prompts(best, evolution.seeds)
    scores = score_test_set(tester, prompts)
    log.test(prompts, scores)
    print("\ntest set (partial credit):")
    for name, score in scores.items():
        print(f"  {name:<10} {score:.3f}")


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
    )
    solver_model = solver(args.solver)
    dataset = DATASETS[args.dataset](args.items, args.data_dir)
    log = RunLog(log_path)
    evolution = Evolution(
        config,
        Evaluator(dataset, solver_model, args.signal),
        LLMOperators(mutator(args.mutator), args.feedback, args.seed),
        SELECTORS[args.selection](),
        log,
    )
    population = evolution.run(SEED_PROMPTS)

    best = population[0]
    print(f"\nbest prompt (train score {best.evaluation.score:.3f}):")
    print(best.prompt)
    if args.held_out:
        _held_out(args, solver_model, evolution, best, log)
    print(f"log: {log_path}")
