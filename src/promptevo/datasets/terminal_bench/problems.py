"""What can go wrong in a Terminal-Bench run."""

from enum import StrEnum
from typing import NamedTuple


# enum for what went wrong in a run. empty means all fine
class Status(StrEnum):
    BAD_JSON     = "bad json"         # usually output got cut off
    NO_FILES     = "no files"
    UNKNOWN_PATH = "unknown path"
    SYNTAX_ERROR = "syntax error"
    TIMEOUT      = "tests timed out"
    NO_RESULTS   = "no test results"  # anti-cheat check failed or crash


# one thing that went wrong. path is the related file (if any)
class Problem(NamedTuple):
    status: Status
    path:   str = ""

    def __str__(self):
        return f"{self.status} {self.path}".strip()


def format_problems(problems: list[Problem]) -> str:
    return ", ".join(map(str, problems)) or "ok"
