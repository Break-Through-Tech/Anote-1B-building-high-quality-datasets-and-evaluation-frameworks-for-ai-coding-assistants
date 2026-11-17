"""Helpers that turn model replies into prompts and code, and score items."""

import pytest

from promptevo.datasets.base import ItemResult, RepairResult
from promptevo.datasets.human_eval.dataset import _imports, _last_block
from promptevo.scoring.evaluation import binary, naive, partial
from promptevo.search.mutations import _clean, _distinct, _first_item
from promptevo.unused.terminal_bench.inject_bugs import _mutate, _sites
from promptevo.unused.terminal_bench.reply import _parse_reply


@pytest.mark.parametrize("reply, prompt", [
    ("New prompt: Be careful.", "Be careful."),
    ("INSTRUCTION MUTANT: Check edge cases.", "Check edge cases."),
    ("instruction:   Check edge cases.", "Check edge cases."),
    ("1. Read the docstring first.", "Read the docstring first."),
    ('"Quoted prompt."', "Quoted prompt."),
    ("Plain prompt.", "Plain prompt."),
])
def test_clean_strips_labels(reply, prompt):
    assert _clean(reply) == prompt


def test_first_item_skips_the_preamble():
    reply = "Here are 100 hints:\n\n1. Read the tests.\n2. Keep it short."
    assert _first_item(reply) == "Read the tests."
    assert _first_item("- Use types.\n- Be brief.") == "Use types."
    assert _first_item("Only a preamble:") == ""


def test_distinct_drops_near_duplicates():
    prompts = [
        "Fix the bug carefully and check every edge case.",
        "Fix the bug carefully and check every edge case!",
        "Write tests first.",
    ]
    assert _distinct(prompts) == [prompts[0], prompts[2]]


def test_last_block_prefers_the_one_defining_the_function():
    reply = (
        "The bug is the comparison.\n```python\nx = 1\n```\n"
        "Fixed:\n```python\ndef add(a, b):\n    return a + b\n```\n"
        "```\nprint(add(1, 2))\n```"
    )
    assert _last_block(reply, "add") == "def add(a, b):\n    return a + b"
    assert _last_block("```py\ny = 2\n```", "add") == "y = 2"
    assert _last_block("  no fences  ", "add") == "no fences"


def test_imports_come_from_the_declaration():
    item = {"declaration": "from typing import List\nimport math\n\ndef f():"}
    assert _imports(item) == "from typing import List\nimport math\n\n"


def test_parse_reply_takes_the_path_from_the_block_or_before_it():
    paths = ["/app/main.py", "/app/util.py"]
    reply = (
        "### /app/util.py\n```python\nA = 1\n```\n"
        "```python\n# /app/main.py\nB = 2\n```\n"
        "```python\nC = 3\n```"
    )
    # the last block names no path, in it or since the block before, so
    # it's dropped
    assert _parse_reply(reply, paths) == [
        ("/app/util.py", "A = 1\n"),
        ("/app/main.py", "B = 2\n"),
    ]


def test_bug_sites_and_mutation():
    code = "def f(a, b):\n    return min(a, b) + 1 if a < b else True\n"
    kinds = sorted(site.kind for site in _sites("/f.py", code))
    assert kinds == ["arith", "bool", "cmp", "const", "minmax"]
    site = next(s for s in _sites("/f.py", code) if s.kind == "cmp")
    assert "a <= b" in _mutate(code, site)


def test_scores():
    fixed = RepairResult(
        tests={"t1": True, "t2": True, "t3": False},
        to_fix=frozenset({"t1", "t3"}),
        passing=frozenset({"t2"}),
    )
    assert partial(fixed) == 0.5
    assert binary(fixed) == 0.0
    assert naive(fixed) == pytest.approx(2 / 3)
    broke_one = RepairResult(
        tests={"t1": True, "t2": False},
        to_fix=frozenset({"t1"}),
        passing=frozenset({"t2"}),
    )
    assert partial(broke_one) == 0.0
    assert partial(ItemResult(tests={})) == 0.0
    assert binary(ItemResult(tests={"a": True, "b": True})) == 1.0
