"""Results of the study (design.py, runner.py), from its run logs. From
src/:

    python -m promptevo.study.report

Only units where every arm finished count. Each comparison is a difference
inside a unit, averaged over units, with a 95% bootstrap interval over units
and an exact sign-flip permutation p-value, Holm-corrected across the
comparisons (stats.py).
"""

import argparse
import functools
import json
import re

from promptevo.datasets.human_eval.dataset import load_rows
from promptevo.scoring.experiment import Entry
from promptevo.search.runlog import LoggedRun
from promptevo.search.templates import BARE, MUTATE, REFLECT, ROLE
from promptevo.study.design import ARMS, MAIN_ARMS, SOLVER, results
from promptevo.study.stats import holm, mean, paired

# (arm, the arm it's compared against, what the difference tests)
CONTRASTS = [
    ("aware-blind", "naive-blind", "fitness, blind mutation"),
    ("aware-reflect", "naive-reflect", "fitness, reflective mutation"),
    ("naive-reflect", "naive-blind", "mutation, naive fitness"),
    ("aware-reflect", "aware-blind", "mutation, aware fitness"),
    ("aware-reflect-tests", "aware-reflect", "showing failing asserts"),
]


def load(
    arms: list[str],
    only: list[int] | None = None,
    solver: str = SOLVER,
) -> dict[int, dict[str, dict]]:
    """{unit: {arm: summary}} for units where every arm has test scores.
    only keeps just those units, solver picks whose study."""
    units = {}
    for folder in sorted(results(solver).glob("unit*")):
        if not (folder / "split.json").exists():
            continue
        unit = int(folder.name[4:])
        if only is not None and unit not in only:
            continue
        split = json.loads((folder / "split.json").read_text())
        summaries = {}
        for arm in arms:
            run = LoggedRun(folder / f"{arm}.jsonl")
            # a run still going or cut off mid-line has no test scores yet
            if run.finished:
                summaries[arm] = _summary(run, split)
        if len(summaries) == len(arms):
            units[unit] = summaries
    return units


def _summary(run, split):
    test = run.test
    picked = test[Entry.VAL_PICK]
    seed = test[Entry.BEST_SEED]
    return {
        "copies": _copy_rate(run.mutations),
        "tried": run.tried,
        "test": picked.score,
        "strict": picked.strict,
        # both picked on the validation set, so this is what the evolved
        # prompts added to the seeds
        "gain": picked.score - seed.score,
        "strict gain": picked.strict - seed.strict,
        # the picked seed can lose to no system prompt at all, so this is
        # the baseline that doesn't depend on picking
        "over no prompt": picked.score - test[Entry.NO_PROMPT].score,
        "evolved": test[Entry.EVOLVED].score,
        "changed": picked.prompt != seed.prompt,
        "train leaks": _leaks(picked.prompt, set(split["train"])),
        "test leaks": _leaks(picked.prompt, set(split["test"])),
        "prompt": picked.prompt,
        "crashed": run.crashed,
    }


def _grams(text, n=6):
    words = re.findall(r"[a-z']+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


# 6-word runs of the mutator's own instructions, placeholders cut out
_TEMPLATES = set().union(*(
    _grams(part)
    for template in [MUTATE, REFLECT, ROLE, BARE]
    for part in re.split(r"\{\w+\}", template)
))


# share of children that picked up a run of the mutator's instructions
# their parent didn't have. small mutators copy gepa's meta-prompt
def _copy_rate(mutations):
    children = [m for m in mutations if m.child]
    if not children:
        return 0.0
    copied = sum(
        bool((_grams(m.child) - _grams(m.parent)) & _TEMPLATES)
        for m in children
    )
    return copied / len(children)


# {task id: row} of humanevalfix, and the function names two problems share
@functools.cache
def _problems():
    rows = load_rows()
    names = [row["entry_point"].lower() for row in rows.values()]
    return rows, {n for n in names if names.count(n) > 1}


# items whose function name or an assert's call shows up in the prompt.
# counted for test items too, as the false-positive rate. only names with
# an underscore count, add or median are also plain words. a name two
# problems share (triangle_area) can't say which one leaked, so those only
# count by their asserts
def _leaks(prompt, items):
    rows, shared = _problems()
    text = prompt.lower()
    found = 0
    for item in items:
        row = rows[item]
        calls = re.findall(r"candidate\(([^\n]{4,}?)\)\s*[=!<>]", row["test"])
        name = row["entry_point"].lower()
        named = "_" in name and name not in shared and name in text
        if named or any(call.lower() in text for call in calls):
            found += 1
    return found


def _print_arms(units, arms):
    print(f"{len(units)} complete units: {sorted(units)}\n")
    print(f"{'arm':<28}{'test':>7}{'strict':>8}{'gain':>8}{'changed':>9}"
          f"{'train leaks':>13}{'test leaks':>12}{'copies':>8}{'tried':>7}")
    for arm in arms:
        rows = [units[u][arm] for u in units]
        print(f"{arm:<28}{mean(r['test'] for r in rows):>7.3f}"
              f"{mean(r['strict'] for r in rows):>8.3f}"
              f"{mean(r['gain'] for r in rows):>+8.3f}"
              f"{sum(r['changed'] for r in rows):>6}/{len(rows):<2}"
              f"{mean(r['train leaks'] for r in rows):>13.2f}"
              f"{mean(r['test leaks'] for r in rows):>12.2f}"
              f"{mean(r['copies'] for r in rows):>8.0%}"
              f"{mean(r['tried'] for r in rows):>7.1f}")
    for unit, summaries in units.items():
        for arm, summary in summaries.items():
            if summary["crashed"]:
                print(f"  unit {unit} {arm}: {summary['crashed']} prompts "
                      "scored 0 because their evaluation crashed")


def _print_contrasts(units, metric):
    print(f"\npaired differences in {metric} (val pick, test set)")
    # with few units even a unanimous result can't be significant, and the
    # interval is just the spread of a handful of numbers
    n = len(units)
    print(f"  {n} units, so the smallest possible p is {2 / 2 ** n:.3f}"
          + (" (intervals mean little below 6 units)" if n < 6 else ""))
    rows = []
    for better, base, what in CONTRASTS:
        diffs = [units[u][better][metric] - units[u][base][metric]
                 for u in units]
        rows.append((better, base, what, *paired(diffs)))
    adjusted = holm([row[-1] for row in rows])
    for (better, base, what, avg, low, high, p), p_holm in zip(
        rows, adjusted
    ):
        print(f"  {better} - {base} ({what}): {avg:+.3f} "
              f"[{low:+.3f}, {high:+.3f}], p {p:.3f}, holm {p_holm:.3f}")


# val pick minus best seed per arm (gain), and minus no system prompt, then
# averaged over the arms inside each unit first ("pooled") when there are
# several
def _print_gains(units, arms):
    print(f"\nval pick - best seed (gain) or - no prompt on the test set, "
          f"{len(units)} units")
    groups = [(arm, [arm]) for arm in arms]
    if len(arms) > 1:
        groups.append(("pooled", arms))
    for metric in ["gain", "strict gain", "over no prompt"]:
        for name, group in groups:
            diffs = [
                mean(units[u][arm][metric] for arm in group) for u in units
            ]
            avg, low, high, p = paired(diffs)
            print(f"  {metric:<16}{name:<24}{avg:+.3f} "
                  f"[{low:+.3f}, {high:+.3f}]  p {p:.4f}  units up "
                  f"{sum(d > 0 for d in diffs)} down "
                  f"{sum(d < 0 for d in diffs)}")


# every other arm minus base, inside each unit
def _print_versus(units, arms, base):
    print(f"\nminus {base} on the test set, {len(units)} units")
    for metric in ["test", "strict"]:
        for arm in arms:
            if arm == base:
                continue
            diffs = [
                units[u][arm][metric] - units[u][base][metric] for u in units
            ]
            avg, low, high, p = paired(diffs)
            print(f"  {metric:<8}{arm:<28}{avg:+.3f} "
                  f"[{low:+.3f}, {high:+.3f}]  p {p:.4f}")


def _parse_args():
    parser = argparse.ArgumentParser(prog="promptevo.study.report")
    parser.add_argument("--prompts", action="store_true",
                        help="print every val pick prompt too")
    # the 5 contrasts only print for the main arms
    parser.add_argument("--arms", nargs="+", choices=list(ARMS),
                        default=MAIN_ARMS)
    parser.add_argument("--units", type=int, nargs="+",
                        help="only these units")
    parser.add_argument("--gain", action="store_true",
                        help="val pick vs best seed and vs no prompt, per "
                        "arm and pooled")
    parser.add_argument("--versus", choices=list(ARMS),
                        help="the other arms minus this one")
    parser.add_argument(
        "--solver", default=SOLVER,
        help="whose study, as given to promptevo.study --solver",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    units = load(args.arms, args.units, args.solver)
    if not units:
        raise SystemExit("no unit has every arm finished yet")
    _print_arms(units, args.arms)
    if len(units) >= 2 and args.arms == MAIN_ARMS:
        for metric in ["test", "strict"]:
            _print_contrasts(units, metric)
    if len(units) >= 2 and args.gain:
        _print_gains(units, args.arms)
    if len(units) >= 2 and args.versus:
        if args.versus not in args.arms:
            raise SystemExit("--versus has to be one of --arms")
        _print_versus(units, args.arms, args.versus)
    if args.prompts:
        for unit, arms in units.items():
            for arm, summary in arms.items():
                print(f"\nunit {unit} {arm}:\n{summary['prompt']}")


if __name__ == "__main__":
    main()
