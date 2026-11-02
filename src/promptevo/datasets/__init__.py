from promptevo.datasets.base import Dataset, ItemResult, RepairResult
from promptevo.datasets.terminal_bench import TerminalBench
from promptevo.datasets.human_eval.dataset import HumanEval
# {--dataset name: Dataset class}. add new ones here
DATASETS = {
    "terminal-bench": TerminalBench,
    "humaneval": HumanEval,
}
