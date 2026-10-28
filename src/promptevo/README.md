# src/promptevo

Evolves the system prompt of a coding model with a (mu+lambda) evolutionary
algorithm. A prompt's fitness comes from giving it to a solver model, letting
the solver attempt each item (fix a bug, write a function), and running the
item's tests.

`Evolution` gets an `Evaluator`, an `Operators` and a `Selector`, so each one
can be swapped without touching the loop. The `Evaluator` is a `Dataset`, a
solver model and a `Signal`.

## How it works

There are two models. The solver is the coding model whose system prompt we're
evolving, it's the one that actually fixes the code. The mutator is the model
that writes new prompts. They can be the same Ollama model or different ones.

The evolutionary run:

1. Score every seed prompt (`SEED_PROMPTS` in `__main__.py`) on the train
   items and keep the best mu of them. That's the starting population.
2. Each generation makes lambda children. For each child the `Selector` picks
   two parents, the mutator combines them (crossover), and with probability
   `--mutation-rate` the mutator also rewrites the result once (mutation).
3. Score the children.
4. Keep the best mu prompts out of parents + children (plus selection), or out
   of the children only with `--comma`.
5. After the last generation, print the best prompt and score it on the
   test set.

## Running it

Docker Desktop and Ollama have to be running first. Run everything from
`src/`.

```bash
cd src

# quick check that everything runs, train and test on the same task
python -m promptevo risk-scorer-replay --held-out risk-scorer-replay

# bug-injected variants (see Terminal-Bench below)
python -m promptevo.datasets.terminal_bench.inject_bugs risk-scorer-replay
python -m promptevo risk-scorer-replay--bug01 risk-scorer-replay--bug02 \
    --held-out other-task--bug01 \
    --data-dir ../data/processed/bug-injected/tasks
```

Positional arguments are the train items. For Terminal-Bench those are task
folder names under `--data-dir`.

## Train and test sets

Same idea as a train/test split in ML. The positional items are the train
set, the only scores the EA ever sees. `--held-out` items are the test set,
scored once at the end. The output looks like this (made-up numbers)

```
test set (partial credit):
  evolved    0.420
  best seed  0.367
  no prompt  0.242
  seed 0     0.313
  ...
```

`best seed` is the seed with the highest train score, which is what you'd get
with 0 generations. So `evolved` vs `best seed` is what the generations added
on top of picking a seed. `no prompt` is the solver with no system prompt at
all. The seeds and these comparisons are in `experiment.py`.

The test set is always scored with partial credit so runs trained with
different `--signal`s are comparable. Keep the two sets separate, and for
bug-injected variants take them from different original tasks.

## Flags

| flag | default | what it does |
|---|---|---|
| `--held-out` | none | the test set, scored only at the end |
| `--dataset` | `terminal-bench` | which entry of `DATASETS` to use |
| `--data-dir` | the dataset's own | folder with the items |
| `--solver` | `qwen2.5-coder:7b` | Ollama model that attempts the items |
| `--mutator` | `qwen2.5-coder:7b` | Ollama model that writes new prompts |
| `--mu` | 2 | population size, how many prompts survive each generation |
| `--lambda` | 2 | children made per generation |
| `--generations` | 2 | how many generations to run |
| `--mutation-rate` | 0.5 | chance a child also gets mutated after crossover |
| `--comma` | off | survivors come from the children only, needs lambda >= mu |
| `--seed` | 0 | random seed for selection, ties and mutation |
| `--selection` | `uniform` | how parents are picked, see Scoring |
| `--signal` | `partial` | how an item's tests turn into a score, see Scoring |
| `--feedback` | `none` | how much the mutator is told, see Scoring |
| `--log` | `results/evolution-<time>.jsonl` | where the run log goes |

A run scores up to `seeds + lambda * generations` prompts, each on every
train item, so it gets slow quickly.

Runs are reproducible, the same `--seed` gives the same run. The solver runs
at temperature 0, and the mutator runs at 0.8 with a new seed for every call
drawn from `--seed`.

## Scoring

Every item comes back with which of its tests passed. `--signal` turns that
into a score between 0 and 1.

- `partial` is the share of the item's target tests that pass. For fix-a-bug
  items the targets are the tests the buggy code failed, and the item scores
  0 if any test that used to pass now fails
- `binary` is 1 only if every target test passes and nothing broke

A prompt's score is the average over the train items. Partial gives the EA
something to climb when the solver fixes some tests but not all of them,
binary is closer to how benchmarks usually report results.

`--selection` decides how parents are picked.

- `uniform` picks two different prompts at random
- `lexicase` looks at individual tests instead of the average. It shuffles
  all the tests, keeps only the prompts that do best on the first one, then
  the next, until one prompt is left. That lets a prompt that's the only one
  fixing some hard test become a parent even if its average is lower

`--feedback` decides what the mutator sees besides the prompt itself.

- `none` means it rewrites blind
- `score` adds the prompt's score
- `failures` adds the score plus, per item, its score, the names of tests
  still failing and any problems (like a syntax error)

## The run log

Every run writes a JSONL file, one JSON object per line, with three kinds of
lines.

```json
{"event": "scored", "generation": 1, "prompt": "...", "score": 0.5,
 "cases": {"task::test_a": 1.0}, "feedback": "...", "crashed": false}
{"event": "generation", "generation": 1,
 "survivors": [{"prompt": "...", "score": 0.5}]}
{"event": "test", "results": {"evolved": {"prompt": "...", "score": 0.42}}}
```

`scored` is written once per unique prompt, `generation` once per generation
with the prompts that survived it. Generation 0 is the seeds. `test` is the
test set at the end.

## Terminal-Bench

`TerminalBench` is the one dataset so far. Each item is a Terminal-Bench task
folder (from `data/raw/terminal-bench/tasks` by default), run in Docker. It
runs the tests once on the untouched code to find which ones the bug breaks,
and returns a `RepairResult` per task.

```bash
# reference solution should score 1, untouched code 0
python -m promptevo.datasets.terminal_bench risk-scorer-replay --mode perfect
python -m promptevo.datasets.terminal_bench risk-scorer-replay --mode nothing

# bug-injected variants, written to data/processed/bug-injected/tasks
python -m promptevo.datasets.terminal_bench.inject_bugs risk-scorer-replay
```

## Using it from Python

The CLI just puts the objects together. `notebooks/run_evolution.ipynb` does
the same thing step by step, and it is probably easier to swap pieces in and
out there.

## Adding a dataset

Subclass `Dataset` and implement `run()`. It gets the prompt and the solver
model and returns `{item name: result}`. Pick the result type by the kind of
item

- `ItemResult(tests)` when the model writes code from scratch (APPS, short
  coding challenges). Every test counts, so `partial` is APPS's "test case
  average" and `binary` is its "strict accuracy"
- `RepairResult(tests, to_fix=..., passing=...)` when the model fixes buggy
  code (HumanEvalFix, Terminal-Bench). Only the tests the bug breaks count,
  and the item scores 0 if a test that used to pass breaks

```python
# src/promptevo/datasets/apps.py
from promptevo.datasets.base import Dataset, ItemResult


class Apps(Dataset):
    def __init__(self, items, data_dir=None):
        self.items = items

    def run(self, prompt, solver):
        results = {}
        for item in self.items:
            code = solver.chat(question_for(item), system=prompt)
            # run code on the item's test cases
            results[item] = ItemResult(
                tests={"case_1": True, "case_2": False},
                problems=[],  # like ["syntax error"]
            )
        return results
```

A repair dataset is the same except for the result

```python
RepairResult(
    tests={"assert_1": True, "assert_2": False},
    to_fix=frozenset({"assert_2"}),    # failed on the buggy code
    passing=frozenset({"assert_1"}),   # passed on the buggy code
)
```

Then add it to `DATASETS` in `datasets/__init__.py`

```python
DATASETS = {
    "terminal-bench": TerminalBench,
    "apps": Apps,
}
```

and pass `--dataset apps`. Scoring and the lexicase stuff all work off the
result, so you don't have to touch anything else. `TerminalBench` in
`datasets/terminal_bench/dataset.py` is a working one to look at.
