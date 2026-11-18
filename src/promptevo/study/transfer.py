"""Scores the study's picked prompts with another solver, to see if what
evolution found with one model helps others. From src/:

    python -m promptevo.study.transfer --solver qwen2.5-coder:3b --units 40 41
    python -m promptevo.study.transfer --report qwen2.5-coder:3b

Every unit gets its own test set and the prompts the study scored there:
no prompt, the seeds, the best seed and each arm's val pick. The solver is
served by vLLM like the study's (see the README). Logs go to
results/transfer/<solver>/unit<n>.jsonl, a run's test event like the
study writes, and a unit that has one is skipped.
"""

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from operator import attrgetter

from promptevo import paths
from promptevo.datasets.human_eval.dataset import HumanEval
from promptevo.llm.models import solver
from promptevo.scoring.evaluation import Evaluator, Signal
from promptevo.scoring.experiment import (
    Entry,
    is_seed_entry,
    score_test_set,
    seed_entry,
)
from promptevo.search.runlog import LoggedRun, RunLog
from promptevo.study.design import RESULTS as STUDY
from promptevo.study.design import SOLVER, safe_name
from promptevo.study.runner import keep_awake
from promptevo.study.stats import holm, paired

RESULTS = paths.RESULTS / "transfer"

# the three operator arms, then rewrite alone as the control
WORKING = ["aware-mixed", "aware-eda", "aware-zero-order-hyper"]
ARMS = [*WORKING, "aware-blind"]

# datasets' load_from_disk isn't thread safe (tqdm's thread_map lock), so
# units build their test sets one at a time
_LOADING = threading.Lock()


def prompts(unit: int) -> dict[str, str]:
    """{name: prompt} the study scored on unit's test set. each arm's val
    pick goes under the arm's name."""
    return {name: r.prompt for name, r in _study_results(unit).items()}


# {name: Tested} from the study's logs of unit. the baselines come from the
# first arm, each arm's val pick goes under its name
def _study_results(unit):
    tests = {
        arm: LoggedRun(STUDY / f"unit{unit}" / f"{arm}.jsonl").test
        for arm in ARMS
    }
    baselines = [
        {n: r for n, r in test.items() if _baseline(n)}
        for test in tests.values()
    ]
    # same seeds, validation set and solver in every arm, so the same
    # baselines down to every reply
    if any(b != baselines[0] for b in baselines):
        raise ValueError(f"unit {unit}'s arms have different baselines")
    return baselines[0] | {
        arm: test[Entry.VAL_PICK] for arm, test in tests.items()
    }


def _baseline(name):
    return name in (Entry.NO_PROMPT, Entry.BEST_SEED) or is_seed_entry(name)


def _folder(name):
    return RESULTS / safe_name(name)


# one unit's test set with every prompt, logged. a failure comes back as
# text, not an exception, so the other units carry on
def _run(unit, name, url):
    try:
        return _score_unit(unit, name, url)
    except Exception as error:
        return f"unit {unit}: FAILED, {error!r}"


def _score_unit(unit, name, url):
    log = _folder(name) / f"unit{unit}.jsonl"
    if LoggedRun(log).finished:
        return f"unit {unit}: already done"
    if log.exists():
        log.rename(log.with_suffix(f".incomplete-{int(time.time())}"))
    start = time.time()
    split = json.loads((STUDY / f"unit{unit}" / "split.json").read_text())
    with _LOADING:
        dataset = HumanEval(
            split["test"], sandbox_cache=STUDY / "sandbox.sqlite"
        )
    model = solver(name, url, cache=RESULTS / "replies.sqlite")
    tester = Evaluator(dataset, model, Signal.PARTIAL)
    named = prompts(unit)
    RunLog(log).test(named, score_test_set(tester, named))
    return f"unit {unit}: done in {(time.time() - start) / 60:.0f} min"


# {unit: {name: Tested}} for a solver, finished units only. the study's own
# solver reads the study logs
def _load(name):
    if name == SOLVER:
        return {
            int(folder.name[4:]): _study_results(int(folder.name[4:]))
            for folder in sorted(STUDY.glob("unit*"))
            if all(
                LoggedRun(folder / f"{arm}.jsonl").finished for arm in ARMS
            )
        }
    runs = [LoggedRun(log) for log in sorted(_folder(name).glob("unit*.jsonl"))]
    return {int(run.path.stem[4:]): run.test for run in runs if run.finished}


# val pick minus a baseline, averaged over the working arms in each unit
def _pooled(unit, baseline, measure):
    return statistics.fmean(
        measure(unit[arm]) - measure(unit[baseline]) for arm in WORKING
    )


_SCORE, _STRICT = attrgetter("score"), attrgetter("strict")
# what every solver's table compares the val picks with
_BASES = [Entry.NO_PROMPT, Entry.BEST_SEED, seed_entry(0)]


def _print_solver(name, units):
    print(f"\n{name}, {len(units)} units")
    units = list(units.values())
    for key in [*_BASES, *ARMS]:
        test = statistics.fmean(u[key].score for u in units)
        strict_share = statistics.fmean(u[key].strict for u in units)
        print(f"  {key:<24}test {test:.3f}  strict {strict_share:.3f}")
    for base in _BASES:
        for kind, measure in [("test", _SCORE), ("strict", _STRICT)]:
            diffs = [_pooled(u, base, measure) for u in units]
            _print_diff(f"{kind:<7}{'working arms - ' + base:<30}", diffs)
    for arm in ARMS:
        diffs = [u[arm].score - u[Entry.NO_PROMPT].score for u in units]
        mean, low, high, p = paired(diffs)
        print(f"  test   {arm + ' - no prompt':<30}{mean:+.3f} "
              f"[{low:+.3f}, {high:+.3f}]  p {p:.4f}")


def _print_diff(label, diffs):
    mean, low, high, p = paired(diffs)
    print(f"  {label}{mean:+.3f} [{low:+.3f}, {high:+.3f}]  p {p:.4f}  up "
          f"{sum(d > 0 for d in diffs)} down {sum(d < 0 for d in diffs)}")


# do the val picks that helped the study's solver most help this one most?
# over every arm's evolved pick, gain over no prompt on each solver
def _print_agreement(name, units, original):
    pairs = [
        (original[u][arm].score - original[u][Entry.NO_PROMPT].score,
         units[u][arm].score - units[u][Entry.NO_PROMPT].score)
        for u in units for arm in ARMS
        if not any(
            units[u][arm].prompt == units[u][n].prompt
            for n in units[u] if is_seed_entry(n)
        )
    ]
    gains = list(zip(*pairs))
    # spearman needs 2 picks and some spread on both sides
    if len(pairs) < 2 or any(len(set(side)) < 2 for side in gains):
        print(f"  spearman vs {SOLVER}: too few evolved picks to compare")
        return
    rho = statistics.correlation(*gains, method="ranked")
    print(f"  spearman of gains over no prompt, {SOLVER} vs {name}: "
          f"{rho:+.2f} over {len(pairs)} evolved picks")


def report(names: list[str]):
    """the transfer tables for each solver, the study's solver first."""
    original = _load(SOLVER)
    loaded = {name: _load(name) for name in names}
    shared = set(original).intersection(*map(set, loaded.values()))
    if len(shared) < 2:
        raise SystemExit(
            f"only {len(shared)} units finished for every solver, need 2"
        )
    original = {u: original[u] for u in sorted(shared)}
    _print_solver(SOLVER, original)
    primary = []
    for name, units in loaded.items():
        units = {u: units[u] for u in sorted(shared)}
        _print_solver(name, units)
        _print_agreement(name, units, original)
        diffs = [_pooled(u, Entry.NO_PROMPT, _SCORE) for u in units.values()]
        primary.append((name, *paired(diffs)))
    print("\nworking arms - no prompt on the test set, holm across solvers")
    for (name, mean, low, high, p), adjusted in zip(
        primary, holm([row[-1] for row in primary])
    ):
        print(f"  {name:<28}{mean:+.3f} [{low:+.3f}, {high:+.3f}]  "
              f"p {p:.4f}  holm {adjusted:.4f}")


def _parse_args():
    parser = argparse.ArgumentParser(prog="promptevo.study.transfer")
    parser.add_argument("--solver", help="the vllm --served-model-name")
    parser.add_argument("--units", type=int, nargs="+")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--report", nargs="+", metavar="SOLVER",
                        help="print the tables for these solvers instead")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.report:
        report(args.report)
        return
    if not (args.solver and args.units):
        raise SystemExit("--solver and --units, or --report")
    keep_awake()
    with ThreadPoolExecutor(args.parallel) as pool:
        jobs = [
            pool.submit(_run, unit, args.solver, args.vllm_url)
            for unit in args.units
        ]
        for job in as_completed(jobs):
            print(job.result(), flush=True)


if __name__ == "__main__":
    main()
