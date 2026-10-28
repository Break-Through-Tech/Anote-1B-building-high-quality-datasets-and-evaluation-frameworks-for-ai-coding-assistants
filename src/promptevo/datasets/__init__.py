from promptevo.datasets.base import Dataset, ItemResult, RepairResult
from promptevo.datasets.terminal_bench import TerminalBench

# {--dataset name: Dataset class}. add new ones here
DATASETS = {
    "terminal-bench": TerminalBench,
}
