from datasets import load_dataset
from promptevo.datasets.base import Dataset
from datasets import load_from_disk
import ast
from promptevo.models import OllamaModel
from promptevo.datasets.base import RepairResult


class HumanEval(Dataset):
    def __init__(self, items, data_dir=None):
        ds = load_dataset("bigcode/humanevalpack", "python")["test"]
        ds.save_to_disk("src/promptevo/datasets/humanevalpack")
        self.items = load_from_disk("src/promptevo/datasets/humanevalpack")

    # extract asserts from the field through ast tree
    def split_asserts(self, code):
        tree = ast.parse(code)
        asserts = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                asserts.append(ast.get_source_segment(code, node))
        return asserts
    # identify passing cases
    def passing_cases(self, assertions, code, entry_point):
        namespace={}
        exec(code, namespace)
        namespace["candidate"] = namespace[entry_point]
        passing = {}
        for assertion in assertions:
            try:
                exec(assertion, namespace)
                passing[assertion] = True
            except Exception:
                passing[assertion] = False
        return passing


                
    def run(self, prompt: str, solver: OllamaModel) -> dict[str, RepairResult]:
        results = {}

        randomints = [5, 100, 32, 70]
        for item in self.items.select(randomints):
            buggy_code = item["prompt"] + item["buggy_solution"]

            code = solver.chat(
            item["instruction"]
            + "\n\nHere is the buggy code:\n"
            + buggy_code
            + "\n\nReturn only the complete corrected Python function. No explanations or markdown fences. Make sure to close proper resources.",
            system=prompt
            )
            code = code.strip()

            if "```" in code:
                parts = code.split("```")
                code = parts[1]
                if code.startswith("python"):
                    code = code[len("python"):]
                code = code.strip()
            #evaluate asserts
            asserts = self.split_asserts(item["test"])

            original_results = self.passing_cases(asserts, buggy_code, item["entry_point"])

            repaired_passing = self.passing_cases(asserts, code, item["entry_point"])

            
            results[item["task_id"]] = RepairResult(
                tests=repaired_passing,
                to_fix = frozenset(
                    a for a, passed in original_results.items() if not passed
                ),
                passing = frozenset(
                    a for a, passed in original_results.items() if passed
                )
            
            )
        return results