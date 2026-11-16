"""HumanEvalFix (Python) from bigcode/humanevalpack as a Dataset.

Items are task ids like "Python/0". Each one is a HumanEval function with a
human-written bug. The solver sees the instruction and the buggy code, never
the tests, and the tests run in Docker (sandbox.py).
"""

import ast
import re
from pathlib import Path

from datasets import load_dataset, load_from_disk

from promptevo.datasets.base import Dataset, RepairResult
from promptevo.datasets.human_eval.sandbox import ResultCache, run_jobs
from promptevo.models import ChatModel

# gitignored, so the download isn't committed
DATA_DIR = Path(__file__).parents[4] / "data/raw/humanevalpack"

# {--reply-format: what the end of the user message asks for}. code wants
# the bare function. the others let the solver write before the code, which
# then comes from the last fenced block. open asks for nothing, so the
# system prompt decides
FORMATS = {
    "code": (
        "Return only the complete corrected Python function. "
        "No explanations or markdown fences. "
        "Make sure to close proper resources."
    ),
    "explain": (
        "First explain the bug in one sentence. Then give the complete "
        "corrected Python function in a ```python block."
    ),
    "reason": (
        "Think step by step about what the function should do and where the "
        "bug is. Then give the complete corrected Python function in a "
        "```python block."
    ),
    "open": "",
}


class HumanEval(Dataset):
    """humanevalfix items by task id."""

    description = (
        "Fix the bug in a Python function. The assistant gets the function's "
        "description and its buggy code, and has to reply with the complete "
        "corrected function. Hidden unit tests check the result."
    )

    def __init__(
        self, items, data_dir=None, sandbox_cache=None, reply_format="code"
    ):
        if reply_format not in FORMATS:
            raise ValueError(f"reply_format has to be one of {list(FORMATS)}")
        self.reply_format = reply_format
        path = Path(data_dir or DATA_DIR)
        # download once, later runs load from disk
        if not path.exists():
            ds = load_dataset("bigcode/humanevalpack", "python")["test"]
            ds.save_to_disk(path)
        rows = {row["task_id"]: row for row in load_from_disk(path)}
        self.items = [rows[item] for item in items]
        # {task id: the buggy code's results}. they never change, so the
        # buggy code only runs once
        self._original = {}
        # sqlite file of sandbox results shared between runs, see sandbox.py
        self._cache = ResultCache(sandbox_cache) if sandbox_cache else None

    def __len__(self):
        return len(self.items)

    # extract asserts from the field through ast tree. top-level statements
    # of check() only, so an assert inside a loop keeps its loop. setup
    # lines (imports, test inputs) go in front of the next assert
    def split_asserts(self, code):
        tree = ast.parse(code)
        asserts = []

        check = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "check"
        )
        setup = ""
        for node in check.body:
            if any(isinstance(n, ast.Assert) for n in ast.walk(node)):
                asserts.append(setup + ast.unparse(node))
                setup = ""
            else:
                setup += ast.unparse(node) + "\n"
        return asserts

    # identify passing cases. jobs is {key: (assertions, code, entry_point)},
    # all run together in docker (sandbox.py has the exec part)
    def passing_cases(self, jobs):
        return run_jobs(jobs, self._cache)

    def run(
        self,
        prompt: str,
        solver: ChatModel,
        items: list[str] | None = None,
    ) -> dict[str, RepairResult]:
        results = {}
        jobs = {}
        chosen = [
            item for item in self.items
            if items is None or item["task_id"] in items
        ]
        messages = {}
        tail = FORMATS[self.reply_format]
        for item in chosen:
            buggy_code = item["prompt"] + item["buggy_solution"]

            messages[item["task_id"]] = (
                item["instruction"]
                + "\n\nHere is the buggy code:\n"
                + buggy_code
                + (f"\n\n{tail}" if tail else "")
            )
        # all at once so a vllm solver can batch them
        replies = dict(zip(
            messages, solver.chat_many(list(messages.values()), prompt)
        ))

        for item in chosen:
            buggy_code = item["prompt"] + item["buggy_solution"]
            code = replies[item["task_id"]].strip()

            if self.reply_format != "code":
                code = _last_block(code, item["entry_point"])
            elif "```" in code:
                parts = code.split("```")
                code = parts[1]
                if code.startswith("python"):
                    code = code[len("python"):]
                code = code.strip()
            # models often drop imports like `from typing import List`
            code = _imports(item) + code

            # evaluate asserts
            asserts = self.split_asserts(item["test"])
            entry = item["entry_point"]
            if item["task_id"] not in self._original:
                jobs[item["task_id"], "original"] = (
                    asserts, buggy_code, entry
                )
            jobs[item["task_id"], "repaired"] = (asserts, code, entry)

        outcomes = self.passing_cases(jobs)
        for (task_id, kind), (passing, _) in outcomes.items():
            if kind == "original":
                self._original[task_id] = passing
        for item in chosen:
            original_results = self._original[item["task_id"]]
            repaired_passing, error = outcomes[item["task_id"], "repaired"]

            results[item["task_id"]] = RepairResult(
                tests=repaired_passing,
                problems=[f"crashed ({error})"] if error else [],
                inputs=messages[item["task_id"]],
                output=replies[item["task_id"]],
                to_fix=frozenset(
                    a for a, passed in original_results.items() if not passed
                ),
                passing=frozenset(
                    a for a, passed in original_results.items() if passed
                ),
            )
        return results


# the last fenced block that defines the function, since reasoning comes
# first. any last block if none does, the whole reply if there are none
def _last_block(reply, entry_point):
    blocks = re.findall(r"```[\w+-]*[ \t]*\n(.*?)```", reply, re.DOTALL)
    defining = [b for b in blocks if f"def {entry_point}" in b]
    if defining or blocks:
        return (defining or blocks)[-1].strip()
    return reply.strip()


# the declaration's import lines
def _imports(item):
    lines = item["declaration"].splitlines()
    kept = [line for line in lines if line.startswith(("import ", "from "))]
    return "\n".join(kept) + "\n\n"
