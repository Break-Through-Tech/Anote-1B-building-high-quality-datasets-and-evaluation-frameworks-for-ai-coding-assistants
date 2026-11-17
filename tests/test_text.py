"""Everything sent to a model, compared with what the study's runs sent.

The study's replies are cached by request, and the sandbox's results by
job, so any change here changes results. These tests stop that from
happening by accident.
"""

import json

import httpx
import pytest

from conftest import RecordingModel, attempt, check_golden, individual
from promptevo.llm import models
from promptevo.datasets.base import Dataset
from promptevo.datasets.human_eval import sandbox
from promptevo.datasets.human_eval.dataset import DATA_DIR, FORMATS, HumanEval
from promptevo.search.feedback import Feedback, item_feedback
from promptevo.search.mutations import MUTATIONS, MixedOperators
from promptevo.search.operators import LLMOperators
from promptevo.search.templates import (
    BARE,
    CROSSOVER,
    MUTATE,
    MUTATION_PROMPTS,
    REFLECT,
    ROLE,
    SEED_PROMPTS,
    THINKING_STYLES,
)
from promptevo.unused.terminal_bench import reply as terminal_reply
from promptevo.unused.terminal_bench.inject_bugs import INSTRUCTION

DESCRIPTION = "Fix the bug in a Python function."


def test_fixed_text():
    check_golden("fixed-text", {
        "seeds": SEED_PROMPTS,
        "formats": FORMATS,
        "humaneval": HumanEval.description,
        "dataset": Dataset.description,
        "mutate": MUTATE,
        "crossover": CROSSOVER,
        "bare": BARE,
        "role": ROLE,
        "reflect": REFLECT,
        "mutation prompts": MUTATION_PROMPTS,
        "thinking styles": THINKING_STYLES,
        "regression": INSTRUCTION,
    })


# the seeds' population, one of them solves item b fully for lamarckian
def _population():
    return [
        individual("Seed A", 0.6),
        individual("Seed B", 0.4),
        individual("Seed C", 0.0),
    ]


def _mutate_six(names, show_tests=True):
    model = RecordingModel()
    ops = MixedOperators(
        model,
        [MUTATIONS[name]() for name in names],
        DESCRIPTION,
        seed=3,
        show_tests=show_tests,
    )
    population = _population()
    ops.observe(population)
    children = []
    for i in range(6):
        parent = population[i % 3]
        children.append(ops.mutate(parent.prompt, parent.evaluation))
    # a crossover child inherits its parent's mutation prompt
    children.append(ops.crossover(children[0], "Seed B"))
    children.append(ops.mutate(children[-1], population[0].evaluation))
    return {"sent": model.sent, "children": children}


def test_every_mutation_sends_the_same_text():
    check_golden("mutations", {
        name: _mutate_six([name]) for name in MUTATIONS
    })


def test_mixed_operators_draw_in_the_same_order():
    mixed = ["rewrite", "first-order", "eda", "zero-order-hyper"]
    check_golden("mixed-operators", {
        "mixed": _mutate_six(mixed),
        "reflective hidden tests": _mutate_six(["reflective"], False),
    })


def test_code_from_the_mutator_keeps_the_parent():
    model = RecordingModel(["```python\nprint(1)\n```"])
    ops = MixedOperators(model, [MUTATIONS["eda"]()], DESCRIPTION)
    ops.observe(_population())
    assert ops.mutate("Seed A", _population()[0].evaluation) == ""


def test_llm_operators_send_the_same_text():
    sent = {}
    for feedback in Feedback:
        model = RecordingModel()
        ops = LLMOperators(model, feedback, seed=5)
        parent = individual("Seed A", 0.6)
        ops.mutate(parent.prompt, parent.evaluation)
        ops.crossover("Seed A", "Seed B")
        sent[feedback.value] = model.sent
    check_golden("llm-operators", sent)


# canned solver replies, one per reply format the dataset parses
_REPLIES = [
    "def f():\n    return 1",
    "Here:\n```python\ndef has_close_elements(n, t):\n    return True\n```",
    "The bug is the sign.\n```python\nimport math\n```\n```python\n"
    "def maximum(arr, k):\n    return sorted(arr)[-k:]\n```",
]
_ITEMS = ["Python/0", "Python/38", "Python/120"]


@pytest.mark.skipif(not DATA_DIR.exists(), reason="humanevalpack not here")
@pytest.mark.parametrize("reply_format", list(FORMATS))
def test_humaneval_sends_the_same_text_and_jobs(reply_format):
    dataset = HumanEval(_ITEMS, reply_format=reply_format)
    jobs = {}

    # the buggy code fails its first assert, every reply passes them all
    def passing_cases(batch):
        out = {}
        for (task, kind), (asserts, code, entry) in batch.items():
            jobs[f"{task} {kind}"] = [asserts, code, entry]
            passing = {a: kind == "repaired" or i > 0
                       for i, a in enumerate(asserts)}
            out[task, kind] = (passing, None)
        return out

    dataset.passing_cases = passing_cases
    solver = RecordingModel(list(_REPLIES))
    results = dataset.run("Be careful.", solver)
    check_golden(f"humaneval-{reply_format}", {
        "sent": solver.sent,
        "jobs": jobs,
        "results": {
            task: {
                "tests": r.tests,
                "to_fix": sorted(r.to_fix),
                "passing": sorted(r.passing),
                "problems": r.problems,
                "inputs": r.inputs,
                "output": r.output,
            }
            for task, r in results.items()
        },
    })


def test_terminal_bench_message(tmp_path):
    host = tmp_path / "main.py"
    host.write_text("# harbor-canary abc\nx = 1\n", encoding="utf-8")
    message = terminal_reply.build_message(
        "Fix the scorer.",
        {"test_one": "AssertionError:   expected 2\n got 1"},
        {"/app/main.py": host},
    )
    check_golden("terminal-bench-message", {"message": message})


def test_cache_keys():
    job = (["assert candidate(1) == 2"], "def f(x):\n    return x", "f")
    request = {
        "served": {"root": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "len": 8192},
        "model": "qwen2.5-coder:1.5b",
        "messages": [
            {"role": "system", "content": "Be careful."},
            {"role": "user", "content": "Fix it. ü"},
        ],
        "temperature": 0,
        "repetition_penalty": 1.0,
        "seed": 0,
        "max_tokens": 1024,
    }
    check_golden("cache-keys", {
        "sandbox job": sandbox._hash(job),
        "reply": models._key(request),
    })


def test_vllm_requests(monkeypatch, tmp_path):
    bodies = []

    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{
                "id": "qwen2.5-coder:1.5b",
                "root": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                "max_model_len": 8192,
            }]})
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": f"reply {len(bodies)}"}}]
        })

    real_client = httpx.Client
    monkeypatch.setattr(
        models.httpx,
        "Client",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    cache = tmp_path / "replies.sqlite"
    model = models.solver(
        "qwen2.5-coder:1.5b", "http://localhost:8000/", cache
    )
    first = model.chat_many(["fix a", "fix b"], "Be careful.")
    again = models.solver(
        "qwen2.5-coder:1.5b", "http://localhost:8000", cache
    ).chat_many(["fix a", "fix b"], "Be careful.")
    assert first == again
    # the second solver read both from the cache
    assert len(bodies) == 2
    check_golden("vllm-requests", {"bodies": bodies, "replies": first})


def test_ollama_options(monkeypatch):
    sent = []

    class FakeClient:
        def __init__(self, timeout):
            sent.append({"timeout": timeout})

        def chat(self, model, messages, options):
            sent.append(
                {"model": model, "messages": messages, "options": options}
            )
            return type("Reply", (), {
                "message": type("Message", (), {"content": "ok"})
            })

    monkeypatch.setattr(models.ollama, "Client", FakeClient)
    models.solver("qwen2.5-coder:7b").chat("fix", system="Be careful.")
    models.mutator("qwen2.5-coder:7b").chat("mutate", seed=11)
    check_golden("ollama-requests", sent)


def test_feedback_shown_to_reflective_mutation():
    result = attempt("a", 0.0)
    check_golden("item-feedback", {
        "shown": item_feedback(result),
        "hidden": item_feedback(result, show_tests=False),
    })
