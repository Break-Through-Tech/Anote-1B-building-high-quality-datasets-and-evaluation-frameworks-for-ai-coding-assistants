"""Results of the study (study.py), from its run logs. From src/:

    python -m promptevo.study_report

Only units where every arm finished count. Each comparison is a difference
inside a unit, averaged over units, with a 95% bootstrap interval over units
and an exact sign-flip permutation p-value, Holm-corrected across the
comparisons.
"""

import argparse
import bisect
import json
import random
import re
from pathlib import Path

from promptevo.datasets.human_eval.dataset import DATA_DIR
from promptevo.mutations import BARE, REFLECT, ROLE
from promptevo.operators import MUTATE
from promptevo.study import ARMS, MAIN_ARMS, SOLVER, finished, results

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
            log = folder / f"{arm}.jsonl"
            # a run still going or cut off mid-line has no test scores yet
            if finished(log):
                summary = _summary(log, split)
                if summary:
                    summaries[arm] = summary
        if len(summaries) == len(arms):
            units[unit] = summaries
    return units


def _summary(log, split):
    with log.open(encoding="utf-8") as f:
        events = [json.loads(line) for line in f]
    test = next((e for e in events if e["event"] == "test"), None)
    if test is None:
        return None
    results = test["results"]
    picked = results["val pick"]
    seed = results["best seed"]
    train_items = set(split["train"])
    return {
        "copies": _copy_rate(events),
        # prompts the search actually tried. a mutation that returns the
        # parent, nothing, or too much text adds none
        "tried": sum(
            e["event"] == "scored" and e["generation"] > 0 for e in events
        ),
        "test": picked["score"],
        "strict": strict(picked),
        # both picked on the validation set, so this is what the evolved
        # prompts added to the seeds
        "gain": picked["score"] - seed["score"],
        "strict gain": strict(picked) - strict(seed),
        # the picked seed can lose to no system prompt at all, so this is
        # the baseline that doesn't depend on picking
        "over no prompt": picked["score"] - results["no prompt"]["score"],
        "evolved": results["evolved"]["score"],
        "changed": picked["prompt"] != seed["prompt"],
        "train leaks": _leaks(picked["prompt"], train_items),
        "test leaks": _leaks(picked["prompt"], set(split["test"])),
        "prompt": picked["prompt"],
        # prompts scored 0 because the solver or sandbox failed, not the
        # prompt
        "crashed": sum(
            e["event"] == "scored" and e.get("crashed", False) for e in events
        ),
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
def _copy_rate(events):
    children = [
        e for e in events if e["event"] == "mutation" and e["child"]
    ]
    if not children:
        return 0.0
    copied = sum(
        bool((_grams(e["child"]) - _grams(e["parent"])) & _TEMPLATES)
        for e in children
    )
    return copied / len(children)


_ROWS = None
_SHARED_NAMES = set()


# train items whose function name or an assert's call shows up in the
# prompt. the same count for test items is the false-positive rate. only
# names with an underscore count, add or median are also plain words. names
# two problems share (triangle_area) can't say which one leaked, so those
# only count by their asserts
def _leaks(prompt, items):
    global _ROWS
    if _ROWS is None:
        from datasets import load_from_disk
        _ROWS = {row["task_id"]: row for row in load_from_disk(DATA_DIR)}
        names = [row["entry_point"].lower() for row in _ROWS.values()]
        _SHARED_NAMES.update(n for n in names if names.count(n) > 1)
    text = prompt.lower()
    found = 0
    for item in items:
        row = _ROWS[item]
        calls = re.findall(r"candidate\(([^\n]{4,}?)\)\s*[=!<>]", row["test"])
        name = row["entry_point"].lower()
        named = "_" in name and name not in _SHARED_NAMES and name in text
        if named or any(call.lower() in text for call in calls):
            found += 1
    return found


def _mean(values):
    values = list(values)
    return sum(values) / len(values)


def strict(test_result: dict) -> float:
    """share of a test result's items with every assert passing."""
    return _mean(i["binary"] for i in test_result["items"].values())


def paired(
    diffs: list[float], seed: int = 0
) -> tuple[float, float, float, float]:
    """mean of the differences, its 95% bootstrap interval and the
    two-sided sign-flip p-value, exact up to 40 units."""
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(
        _mean(rng.choice(diffs) for _ in range(n)) for _ in range(10000)
    )
    return _mean(diffs), means[250], means[9749], _sign_flip_p(diffs)


# every way to flip the signs, 2^n of them, without listing them all. each
# half's 2^(n/2) sums get paired up instead, so 40 units takes seconds.
# past 40 that blows up, so random flips stand in
def _sign_flip_p(diffs):
    if len(diffs) > 40:
        return _sampled_sign_flip_p(diffs)
    half = len(diffs) // 2
    left = _flip_sums(diffs[:half])
    right = sorted(_flip_sums(diffs[half:]))
    # as far from 0 as the observed sum or further. the slack keeps exact
    # ties that float rounding would split
    edge = abs(sum(diffs)) - 1e-9
    if edge <= 0:
        return 1.0
    extreme = sum(
        len(right) - bisect.bisect_left(right, edge - s)
        + bisect.bisect_right(right, -edge - s)
        for s in left
    )
    return extreme / (len(left) * len(right))


# share of random sign flips at least as extreme. the +1 counts the observed
# one, so p is never 0
def _sampled_sign_flip_p(diffs, draws=200000, seed=0):
    rng = random.Random(seed)
    edge = abs(sum(diffs)) - 1e-9
    if edge <= 0:
        return 1.0
    extreme = 0
    for _ in range(draws):
        bits = rng.getrandbits(len(diffs))
        flipped = sum(
            d if bits >> i & 1 else -d for i, d in enumerate(diffs)
        )
        extreme += abs(flipped) >= edge
    return (extreme + 1) / (draws + 1)


def _flip_sums(values):
    sums = [0.0]
    for value in values:
        sums = [s + value for s in sums] + [s - value for s in sums]
    return sums


def holm(pvalues: list[float]) -> list[float]:
    """holm-adjusted p-values, same order."""
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def _print_arms(units, arms):
    print(f"{len(units)} complete units: {sorted(units)}\n")
    print(f"{'arm':<28}{'test':>7}{'strict':>8}{'gain':>8}{'changed':>9}"
          f"{'train leaks':>13}{'test leaks':>12}{'copies':>8}{'tried':>7}")
    for arm in arms:
        rows = [units[u][arm] for u in units]
        print(f"{arm:<28}{_mean(r['test'] for r in rows):>7.3f}"
              f"{_mean(r['strict'] for r in rows):>8.3f}"
              f"{_mean(r['gain'] for r in rows):>+8.3f}"
              f"{sum(r['changed'] for r in rows):>6}/{len(rows):<2}"
              f"{_mean(r['train leaks'] for r in rows):>13.2f}"
              f"{_mean(r['test leaks'] for r in rows):>12.2f}"
              f"{_mean(r['copies'] for r in rows):>8.0%}"
              f"{_mean(r['tried'] for r in rows):>7.1f}")
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
    for (better, base, what, mean, low, high, p), p_holm in zip(
        rows, adjusted
    ):
        print(f"  {better} - {base} ({what}): {mean:+.3f} "
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
                _mean(units[u][arm][metric] for arm in group) for u in units
            ]
            mean, low, high, p = paired(diffs)
            print(f"  {metric:<16}{name:<24}{mean:+.3f} "
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
            mean, low, high, p = paired(diffs)
            print(f"  {metric:<8}{arm:<28}{mean:+.3f} "
                  f"[{low:+.3f}, {high:+.3f}]  p {p:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="promptevo.study_report")
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
    parser.add_argument("--solver", default=SOLVER,
                        help="whose study, as given to study.py --solver")
    args = parser.parse_args()
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
