"""HumanEvalFix (Python) from bigcode/humanevalpack as a Dataset.

Items are task ids like "Python/0". Each one is a HumanEval function with a
human-written bug. The solver sees the instruction and the buggy code, never
the tests, and the tests run in Docker (sandbox.py).
"""

import ast
from pathlib import Path

from datasets import load_dataset, load_from_disk

from promptevo.datasets.base import Dataset, RepairResult
from promptevo.datasets.human_eval.sandbox import run_jobs
from promptevo.models import OllamaModel

# gitignored, so the download isn't committed
DATA_DIR = Path(__file__).parents[4] / "data/raw/humanevalpack"


class HumanEval(Dataset):
    """humanevalfix items by task id."""

    def __init__(self, items, data_dir=None):
        path = Path(data_dir or DATA_DIR)
        # download once, later runs load from disk
        if not path.exists():
            ds = load_dataset("bigcode/humanevalpack", "python")["test"]
            ds.save_to_disk(path)
        rows = {row["task_id"]: row for row in load_from_disk(path)}
        self.items = [rows[item] for item in items]

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
        return run_jobs(jobs)

    def run(self, prompt: str, solver: OllamaModel) -> dict[str, RepairResult]:
        results = {}
        jobs = {}

        for item in self.items:
            buggy_code = item["prompt"] + item["buggy_solution"]

            code = solver.chat(
                item["instruction"]
                + "\n\nHere is the buggy code:\n"
                + buggy_code
                + "\n\nReturn only the complete corrected Python function. "
                "No explanations or markdown fences. "
                "Make sure to close proper resources.",
                system=prompt,
            )
            code = code.strip()

            if "```" in code:
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
            jobs[item["task_id"], "original"] = (asserts, buggy_code, entry)
            jobs[item["task_id"], "repaired"] = (asserts, code, entry)

        outcomes = self.passing_cases(jobs)
        for item in self.items:
            original_results, _ = outcomes[item["task_id"], "original"]
            repaired_passing, error = outcomes[item["task_id"], "repaired"]

            results[item["task_id"]] = RepairResult(
                tests=repaired_passing,
                problems=[f"crashed ({error})"] if error else [],
                to_fix=frozenset(
                    a for a, passed in original_results.items() if not passed
                ),
                passing=frozenset(
                    a for a, passed in original_results.items() if passed
                ),
            )
        return results


# the declaration's import lines
def _imports(item):
    lines = item["declaration"].splitlines()
    kept = [line for line in lines if line.startswith(("import ", "from "))]
    return "\n".join(kept) + "\n\n"
