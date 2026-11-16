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

1. Score every seed prompt (`SEED_PROMPTS` in `experiment.py`) on the train
   items and keep the best mu of them. That's the starting population.
2. Each generation makes lambda children. For each child the `Selector` picks
   two parents, the mutator combines them (crossover), and with probability
   `--mutation-rate` the mutator also rewrites the result once (mutation).
3. Score the children.
4. Keep the best mu prompts out of parents + children (plus selection), or out
   of the children only with `--comma`.
5. After the last generation, print the best prompt, pick one on the
   validation set if there is one, and score them on the test set.

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

### Faster solver with vLLM

The solver can run on a vLLM server instead of Ollama. It's a lot faster,
because vLLM batches all of a prompt's items together. vLLM doesn't run on
Windows, so the server goes in WSL:

```bash
# once, in WSL
uv venv ~/vllm-env --python 3.12 && source ~/vllm-env/bin/activate
uv pip install vllm

# every session. batch invariant makes every reply the same no matter what
# else is in the batch, and costs a bit of speed
VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --served-model-name qwen2.5-coder:1.5b --dtype bfloat16 \
    --max-model-len 8192 --gpu-memory-utilization 0.6 --seed 0 \
    --generation-config vllm --port 8000
```

Then from `src/` on Windows:

```bash
python -m promptevo Python/0 Python/1 --held-out Python/2 \
    --dataset humaneval --solver qwen2.5-coder:1.5b \
    --vllm-url http://localhost:8000
```

Some notes on the setup:

- `VLLM_USE_FLASHINFER_SAMPLER=0` because FlashInfer's sampler compiles
  itself with `nvcc`, which WSL doesn't have, and greedy decoding doesn't
  need it
- lower `--gpu-memory-utilization` if the server doesn't fit next to
  whatever else is on the GPU. The mutator stays on Ollama, which puts what
  fits on the GPU and runs the rest on the CPU
- the client sends every sampling setting itself, so the
  `--generation-config` flag is only a backup. Without either, vLLM uses
  Qwen's 1.1 repetition penalty and the scores change
- replies stop at 1024 tokens. Real fixes are far shorter, and longer
  replies are the model stuck repeating itself
- several runs can share one server, and that's where most of the speed is.
  Batch invariant mode keeps each run's replies the same as if it ran alone

## Train, validation and test sets

Same idea as a train/validation/test split in ML. The positional items are
the train set, the only scores the EA ever sees. `--held-out` items are the
test set, scored once at the end.

`--validation` is optional. The prompt with the best train score is often
just the one that fit those particular items, so with a validation set the
final population and the seeds get scored on it and the best one there is
the `val pick`. If a seed wins, the generations didn't find anything better.

The output looks like this (made-up numbers)

```
test set (partial credit):
  evolved    0.420
  val pick   0.401
  best seed  0.367
  no prompt  0.242
  seed 0     0.313
  ...
```

`evolved` is the best train score. `best seed` is the seed with the highest
train score, which is what you'd get with 0 generations. So `evolved` or
`val pick` vs `best seed` is what the generations added on top of picking a
seed. `no prompt` is the solver with no system prompt at all. The seeds and
these comparisons are in `experiment.py`.

Validation and test are always scored with partial credit so runs trained
with different `--signal`s are comparable. Keep the sets separate, and for
bug-injected variants take them from different original tasks.

## Flags

| flag | default | what it does |
|---|---|---|
| `--validation` | none | picks the final prompt, see above |
| `--held-out` | none | the test set, scored only at the end |
| `--dataset` | `terminal-bench` | which entry of `DATASETS` to use |
| `--data-dir` | the dataset's own | folder with the items |
| `--solver` | `qwen2.5-coder:7b` | model that attempts the items, Ollama's name or vLLM's `--served-model-name` |
| `--mutator` | `qwen2.5-coder:7b` | Ollama model that writes new prompts |
| `--vllm-url` | none | run the solver on this vLLM server, see above |
| `--cache` | none | SQLite file of solver replies that runs share, vLLM only |
| `--sandbox-cache` | none | SQLite file of test results that runs share, HumanEval only. The sandbox gives the same result for the same code every time, so this only saves time |
| `--reply-format` | `code` | what the end of each HumanEval message asks for. `code` wants the bare function, `explain` and `reason` ask for some text before a fenced block, `open` asks for nothing and leaves it to the system prompt |
| `--max-tokens` | 1024 | longest solver reply, vLLM only. The formats that reason first need more |
| `--mu` | 2 | population size, how many prompts survive each generation |
| `--lambda` | 2 | children made per generation |
| `--generations` | 2 | how many generations to run |
| `--mutation-rate` | 0.5 | chance a child also gets mutated after crossover |
| `--crossover-rate` | 1.0 | chance a child starts with crossover, otherwise it's a mutated copy of one parent |
| `--mutations` | `rewrite` | which mutations to use, one is picked at random each time, see Mutations |
| `--screen` | 0 | score children on this many train items first, see Mutations |
| `--search` | `ea` | `ea` is the loop above, `gepa` is GEPA's loop, see GEPA |
| `--budget` | 0 | stop after this many solver calls, 0 = no limit |
| `--comma` | off | survivors come from the children only, needs lambda >= mu |
| `--seed` | 0 | random seed for selection, ties and mutation |
| `--selection` | `uniform`, `pareto` for GEPA | how parents are picked, see Scoring |
| `--signal` | `partial` | how an item's tests turn into a score, see Scoring |
| `--feedback` | `none` | how much the mutator is told, see Scoring |
| `--hide-tests` | off | reflective mutation sees how many tests failed but not the asserts, which hold the expected values |
| `--drift-check` | off | re-scores the best seed at the end with nothing cached (neither cache flag applies to it), a different score means the solver or sandbox drifted |
| `--log` | `results/evolution-<time>.jsonl` | where the run log goes |

A run scores up to `seeds + lambda * generations` prompts, each on every
train item, so it gets slow quickly.

The same `--seed` gives the same run. The solver runs at temperature 0, and
the mutator runs at 0.8 with a new seed for every call drawn from `--seed`.
Scores can still move a little after Ollama restarts, so compare prompts
scored in the same session. A vLLM solver in batch invariant mode gave the
same replies on every rerun, so this only applies to Ollama.
The two backends score differently (Ollama's model is 4-bit), so don't mix
them in one comparison.

## Scoring

Every item comes back with which of its tests passed. `--signal` turns that
into a score between 0 and 1.

- `partial` is the share of the item's target tests that pass. For fix-a-bug
  items the targets are the tests the buggy code failed, and the item scores
  0 if any test that used to pass now fails
- `binary` is 1 only if every target test passes and nothing broke
- `naive` is the share of all the item's tests that pass, the ones the bug
  doesn't touch included. Sending the buggy code back unchanged already
  scores well on it

A prompt's score is the average over the train items. Partial gives the EA
something to climb when the solver fixes some tests but not all of them,
binary is closer to how benchmarks usually report results.

`--selection` decides how parents are picked.

- `uniform` picks two different prompts at random
- `lexicase` looks at individual tests instead of the average. It shuffles
  all the tests, keeps only the prompts that do best on the first one, then
  the next, until one prompt is left. That lets a prompt that's the only one
  fixing some hard test become a parent even if its average is lower
- `pareto` is GEPA's. A prompt that's the best on at least one item can be
  picked, more often the more items it's best on, unless another prompt is
  at least as good on every item
- `roulette` is EvoPrompt's. The chance of being picked is proportional to
  the score
- `lexicase-down` is lexicase on a random quarter of the tests each time
- `greedy` always picks the prompt with the best average

`--selection` defaults to `uniform` for the EA and `pareto` for GEPA. Any of
them works with `--search gepa` too, where they look at the validation
scores.

`--feedback` decides what the mutator sees besides the prompt itself.

- `none` means it rewrites blind
- `score` adds the prompt's score
- `failures` adds the score plus, per item, its score, the names of tests
  still failing and any problems (like a syntax error)

## Mutations

`--mutations` takes any of these, and every mutation picks one at random.
They're in `mutations.py`, with the papers' wording where the papers give
it.

| name | from | what the mutator gets |
|---|---|---|
| `rewrite` | ours | the prompt, make one change |
| `zero-order` | Promptbreeder | the task description only, asked for hints |
| `first-order` | Promptbreeder | the prompt plus a mutation prompt like "Rephrase the instruction without using any of the same words" |
| `eda` | Promptbreeder | the population's prompts as a list to continue |
| `eda-rank` | Promptbreeder | the same list sorted by score |
| `lineage` | Promptbreeder | the best prompt of every generation so far |
| `zero-order-hyper` | Promptbreeder | writes a new mutation prompt from the task and a thinking style, then uses it |
| `first-order-hyper` | Promptbreeder | improves the prompt's mutation prompt, then uses it |
| `lamarckian` | Promptbreeder | one attempt the solver got right, asked what the instruction was |
| `reflective` | GEPA | 3 attempts with what went wrong, asked for a better instruction |

Children that contain a code block are dropped, the 7B mutator sometimes
solves a task instead of writing a prompt.

`--screen 3` is GEPA's minibatch check. A child gets scored on 3 random
train items first, and only on all of them if it beats its parent on those
3. Most children don't, so a run gets through a lot more of them.

## GEPA

`--search gepa` runs GEPA's own loop (`gepa.py`) instead of generations. It
keeps a pool of prompts scored on the validation items. Every step it picks
one with pareto selection, runs it on 3 train items, mutates it with what
happened there, and runs the child on the same 3. A child that scores
higher gets scored on the validation items and joins the pool. It needs
`--validation` and `--budget`, and the budget counts train and validation
calls together.

```bash
python -m promptevo $(cat train.txt) --validation $(cat val.txt) \
    --held-out $(cat test.txt) --dataset humaneval \
    --search gepa --budget 2000 --mutations reflective
```

## The run log

Every run writes a JSONL file, one JSON object per line.

```json
{"event": "scored", "generation": 1, "prompt": "...", "score": 0.5,
 "items": {"task": 0.5}, "cases": {"task::test_a": 1.0}, "feedback": "...",
 "crashed": false}
{"event": "generation", "generation": 1,
 "survivors": [{"prompt": "...", "score": 0.5}], "rollouts": 480}
{"event": "mutation", "kind": "reflective", "parent": "...", "child": "..."}
{"event": "screened", "generation": 1, "prompt": "...", "score": 0.67,
 "parent_score": 0.33}
{"event": "validation", "results": [{"prompt": "...", "score": 0.44}]}
{"event": "test", "results": {"evolved": {"prompt": "...", "score": 0.42}}}
```

`scored` is written once per unique prompt, `generation` once per generation
with the prompts that survived it and the solver calls so far. Generation 0
is the seeds. `mutation` says which mutation made each child, `screened` is
every minibatch check. `validation` and `test` come at the end. For GEPA,
`generation` is the step number.

## HumanEval

`HumanEval` (`--dataset humaneval`) is HumanEvalFix from
`bigcode/humanevalpack`, 164 Python functions with a human-written bug.
Items are task ids like `Python/0`. The data downloads to
`data/raw/humanevalpack` on the first run, and every assert runs in a
`python:3.12-slim` container with no network. Each assert gets 3 s and an
item's asserts get 10 s together, the same total as bigcode's harness for
HumanEvalFix. The total keeps one fix that loops forever from holding up
the whole batch.

```bash
python -m promptevo Python/0 Python/1 --held-out Python/2 Python/3 \
    --dataset humaneval
```

## Terminal-Bench

Each `TerminalBench` item is a Terminal-Bench task
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

## The study

`study.py` runs the fitness x feedback comparison on HumanEvalFix, with the
solver on vLLM (see above). Each unit is a random split into 40 train, 30
validation and 94 test problems, and every arm runs once per unit with the
same split and seed:

| arm | fitness | mutation |
|---|---|---|
| `naive-blind` | naive | rewrite |
| `aware-blind` | partial | rewrite |
| `naive-reflect` | naive | reflective, failing asserts hidden |
| `aware-reflect` | partial | reflective, failing asserts hidden |
| `aware-reflect-tests` | partial | reflective, failing asserts shown |

Naive counts every assert, so sending the buggy code back already scores
well. Partial only counts the asserts the bug breaks (see Scoring).

More arms run only with `--arms`. `binary-blind` selects on binary
fitness. `aware-mixed` is blind, but each child comes from one of rewrite,
first-order, eda or zero-order-hyper, so a mutation can swap the prompt's
whole strategy and not just a phrase. For the ablation, `aware-eda`,
`aware-first-order` and `aware-zero-order-hyper` use one of those alone,
and `mixed-no-<operator>` leaves one out.

```bash
python -m promptevo.study --units 0 1 2 3   # runs, 4 at a time
python -m promptevo.study_report            # tables + paired tests

# val pick vs best seed and vs no prompt on chosen arms and units, per
# arm and pooled
python -m promptevo.study_report --arms naive-blind aware-blind \
    --units 16 17 18 --gain
# the other arms minus one of them
python -m promptevo.study_report --arms aware-blind aware-mixed \
    --versus aware-blind
```

Logs go to `results/study/unit<n>/`. Running it again skips finished runs.
`results/study/study.json` holds the setup, and a run with a different one
stops. New arms can be added later, but an arm that already ran can't
change.
The runs share a cache of solver replies (`results/study/replies.sqlite`)
and one of test results (`results/study/sandbox.sqlite`), so the seeds are
only solved and tested once per problem. The report compares arms
inside each unit, with bootstrap intervals over units and Holm-corrected
sign-flip p-values. It also counts how often an evolved prompt names train
problems (leaks) and how often a mutation copied the mutator's own
instructions (copies).

### Transfer to other solvers

`transfer.py` scores the prompts the study picked with a different solver,
on the same units and test sets. Every pick stays the study solver's, so it
checks whether those prompts help other models too. Start a vLLM server
for the other model as in the vLLM section, with its own
`--served-model-name`, then

```bash
python -m promptevo.transfer --solver qwen2.5-coder:3b --units 40 41 42
# the study's solver first, then each one listed
python -m promptevo.transfer --report qwen2.5-coder:3b
```

Logs go to `results/transfer/<solver>/`, and finished units are skipped.

### Evolving for another solver

`--solver` runs the study itself with a different vLLM model. Its logs,
setup and reply cache go to `results/study-<solver>/`, and the report
takes the same flag. The sandbox cache stays shared, since a test result
doesn't depend on who wrote the code.

```bash
python -m promptevo.study --solver deepseek-coder:1.3b \
    --units 40 41 42 --arms aware-mixed aware-blind
python -m promptevo.study_report --solver deepseek-coder:1.3b \
    --arms aware-blind aware-mixed --gain
```

Check a new model's tokenizer first. DeepSeek-Coder's config names
`LlamaTokenizerFast`, which transformers 5 loads as a different tokenizer
than the model's own `tokenizer.json`, so its replies come back with `Ġ`
for spaces. Give vLLM `--tokenizer` a copy of the model's
`tokenizer.json` and `tokenizer_config.json` whose `tokenizer_class` is
`PreTrainedTokenizerFast`.

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
result, so you don't have to touch anything else. `HumanEval` in
`datasets/human_eval/dataset.py` is a short working one to look at.

## Classes

Indented ones are subclasses of the one above them.

- `Evolution` (`evolution.py`) runs the loop
- `EvolutionConfig` (`evolution.py`) holds mu, lambda, generations and so on
- `RunLog` (`evolution.py`) writes the JSONL log
- `Evaluator` (`evaluation.py`) scores a prompt with a dataset, solver and signal
- `Signal` (`evaluation.py`) is partial, binary or naive credit
- `GEPA` (`gepa.py`) runs GEPA's loop instead of `Evolution`
- `Selector` (`selection.py`) picks parents
  - `UniformSelector`
  - `LexicaseSelector`
    - `DownsampledLexicaseSelector`
  - `ParetoSelector`
  - `RouletteSelector`
  - `GreedySelector`
- `Operators` (`operators.py`) makes new prompts
  - `LLMOperators` asks the mutator model to do mutate and crossover
    - `MixedOperators` (`mutations.py`) picks one of several mutations
- `Mutation` (`mutations.py`) is one way to mutate, one subclass per row of
  the Mutations table
- `Feedback` (`operators.py`) is how much the mutator is told
- `OllamaModel` (`models.py`) is one Ollama model, used for the solver and the
  mutator
- `VLLMModel` (`models.py`) is a model on a vLLM server, for the solver
- `Evaluation`, `Individual` (`types.py`) are a scored prompt and a prompt in
  the population
- `Dataset` (`datasets/base.py`) is a set of items to score prompts on
  - `HumanEval` (`datasets/human_eval/dataset.py`)
  - `TerminalBench` (`datasets/terminal_bench/dataset.py`)
- `ItemResult` (`datasets/base.py`) is one item's tests, all of them count
  - `RepairResult` only counts the tests the bug breaks

Inside `datasets/terminal_bench/` there's also `Task` (one task folder),
`Container` (a Docker container), `BugInjector`, and the small types `Mode`,
`Status`, `Problem`, `RunResult` and `Baseline`.
