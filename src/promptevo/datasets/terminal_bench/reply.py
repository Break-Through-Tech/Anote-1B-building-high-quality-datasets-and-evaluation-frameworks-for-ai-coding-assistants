"""What the solver gets shown, and writing its reply back to the files."""

import ast
import re
import sys
from pathlib import Path

from promptevo.datasets.terminal_bench.problems import Problem, Status

# the "do not train on this" canary also shows up in code files
_CANARY_LINE_RE = re.compile(
    r"^.*(harbor-canary|BENCHMARK DATA SHOULD NEVER).*\n?", re.MULTILINE
)

# only fixed text added to the prompt, keep it neutral.
# fenced code bc 7b models mess up escaping whole files in json
_FORMAT = """
Reply format: you may reason first. Then, for each file you change, write a line
`### <path>` with the exact path shown above, followed by the complete new
contents of that file in a ```python fenced code block. Only include files you change.
"""

_FENCE_RE = re.compile(r"^```[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)


def build_message(
    instruction: str, errors: dict[str, str], files: dict[str, Path]
) -> str:
    """user message: the task, the failing tests, the code, reply format."""
    report = _failure_report(errors)
    shown = "\n\n".join(
        "### {}\n```python\n{}\n```".format(
            path, _CANARY_LINE_RE.sub("", host.read_text(encoding="utf-8"))
        )
        for path, host in files.items()
    )
    return f"{instruction}\n\n{report}\n\n{shown}\n\n{_FORMAT}"


def apply_reply(reply: str, files: dict[str, Path]) -> list[Problem]:
    """writes model's files over the shown ones. returns problems."""
    changes = _parse_reply(reply, list(files))
    if not changes:
        return [Problem(Status.NO_FILES)]
    problems = []
    for path, code in changes:
        try:
            ast.parse(code)
        except SyntaxError as e:
            print(f"{path}: line {e.lineno}: {e.msg}\n{code}", file=sys.stderr)
            problems.append(Problem(Status.SYNTAX_ERROR, path))
        files[path].write_text(code, encoding="utf-8", newline="\n")
    return problems


# failing tests shown like a ci log, so the model knows where to look
def _failure_report(errors):
    lines = ["The test suite currently reports these failures:", ""]
    for name, error in list(errors.items())[:8]:
        # tracebacks can be huge, keep the start of each
        lines += [f"- {name}", "  " + " ".join(error.split())[:600]]
    return "\n".join(lines)


# [(path, code)] per fenced block. path comes from the block's first line
# or the last path mentioned before it, models do both
def _parse_reply(reply, paths):
    changes, prev_end = [], 0
    for m in _FENCE_RE.finditer(reply):
        code = m.group(1)
        first, _, rest = code.partition("\n")
        inside = [p for p in paths if p in first]
        if inside and first.lstrip().startswith("#"):
            path, code = max(inside, key=len), rest
        else:
            before = reply[prev_end : m.start()]
            seen = [(before.rfind(p), len(p), p) for p in paths if p in before]
            path = max(seen)[2] if seen else None
        prev_end = m.end()
        if path:
            changes.append((path, code))
    return changes
