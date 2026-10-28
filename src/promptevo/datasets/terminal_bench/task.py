"""One Terminal-Bench task folder. Task.run() does this:

1. Build the task's Docker images
2. Start the task container, copy its code out
3. Send the model the prompt, the task, the tests the code fails, and the code
4. Write the files the model sends back
5. Put the code in a test container, run the task's tests

Which tests used to pass/fail comes from Task.baseline, the tests run once
on the untouched code.
"""

import functools
import json
import re
import subprocess
import tempfile
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from promptevo.datasets.terminal_bench import reply
from promptevo.datasets.terminal_bench.docker import (
    Container,
    docker,
    image_exists,
    remove_images,
)
from promptevo.datasets.terminal_bench.problems import (
    Problem,
    Status,
    format_problems,
)
from promptevo.models import OllamaModel

# "do not train on this" marker, stripped before the model sees it
CANARY_RE = re.compile(r"<!--[^>]*harbor-canary[^>]*-->\s*", re.IGNORECASE)


# what to do to the code before testing it
class Mode(StrEnum):
    NORMAL  = "normal"   # run the model
    PERFECT = "perfect"  # run the reference solution
    NOTHING = "nothing"  # leave the buggy code as is


class RunResult(NamedTuple):
    tests:    dict[str, bool]  # {test name: passed?}. empty = tests didn't run
    problems: list[Problem]    # empty = ran fine
    errors:   dict[str, str]   # {failed test: its error}


# tests the untouched code fails / passes, and what the failures said
class Baseline(NamedTuple):
    to_fix: frozenset[str]
    keep:   frozenset[str]
    errors: dict[str, str]  # {to_fix test: its error}


class Task:
    """a task folder (task.toml, instruction.md, environment/, tests/,
    solution/). raises on tasks we can't run."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.name = self.path.name
        self.config = tomllib.loads(
            (self.path / "task.toml").read_text(encoding="utf-8")
        )
        compose = self.path / "environment" / "docker-compose.yaml"
        if self.config.get("environment", {}).get("gpus") or compose.exists():
            raise ValueError(
                f"{self.name}: needs gpu or docker-compose, not supported"
            )
        self.task_image = f"tb-env-{self.name}"
        self.test_image = f"tb-test-{self.name}"

    @property
    def instruction(self) -> str:
        raw = (self.path / "instruction.md").read_text(encoding="utf-8")
        return CANARY_RE.sub("", raw).strip()

    # skips existing images, docker build hits docker hub even when cached.
    # delete the images to rebuild
    def build_images(self):
        for image, context in (
            (self.task_image, "environment"),
            (self.test_image, "tests"),
        ):
            if not image_exists(image):
                docker("build", "-q", "-t", image, self.path / context)

    def remove_images(self):
        remove_images(self.task_image, self.test_image)

    # runs the reference solve.sh in the container
    def run_solution(self, container: Container):
        container.copy_to(self.path / "solution", "/solution")
        container.exec("bash", "/solution/solve.sh")

    # copies artifact dirs out into tmp. returns {container path: host dir}
    def copy_out(self, container: Container, tmp: str | Path) -> dict:
        artifacts = {
            path: Path(tmp) / str(i)
            for i, path in enumerate(self.config["artifacts"])
        }
        for path, local in artifacts.items():
            container.copy_from(f"{path.rstrip('/')}/.", local)
        return artifacts

    def run(
        self,
        mode: Mode = Mode.NORMAL,
        prompt: str = "",
        solver: OllamaModel | None = None,
    ) -> RunResult:
        """gets the code into the state mode says, runs the tests."""
        self.build_images()
        problems = []
        with (
            Container(self.task_image) as box,
            tempfile.TemporaryDirectory() as tmp,
        ):
            if mode == Mode.PERFECT:
                self.run_solution(box)
            artifacts = self.copy_out(box, tmp)
            if mode == Mode.NORMAL:
                problems = self._ask_and_apply(prompt, solver, artifacts)
            with Container(self.test_image) as verifier:
                _copy_in(verifier, artifacts)
                run = self._run_tests(verifier)
        return run._replace(problems=problems + run.problems)

    # what tests pass/fail without any code changes. runs once per Task
    @functools.cached_property
    def baseline(self) -> Baseline:
        tests, problems, errors = self.run(Mode.NOTHING)
        to_fix = frozenset(t for t, ok in tests.items() if not ok)
        if not to_fix:
            why = format_problems(problems)
            raise RuntimeError(
                f"{self.path}: untouched code fails no tests ({why})"
            )
        keep = frozenset(t for t, ok in tests.items() if ok)
        return Baseline(to_fix, keep, errors)

    # asks the model, writes its reply. returns problems
    def _ask_and_apply(self, prompt, solver, artifacts):
        files = code_files(artifacts)
        user = reply.build_message(
            self.instruction, self.baseline.errors, files
        )
        return reply.apply_reply(solver.chat(user, system=prompt), files)

    def _run_tests(self, verifier):
        timeout = self.config.get("verifier", {}).get("timeout_sec", 600)
        try:
            verifier.try_exec("bash", "/tests/test.sh", timeout=timeout)
        except subprocess.TimeoutExpired:
            return RunResult({}, [Problem(Status.TIMEOUT)], {})
        # per-test results, pytest writes these
        out = verifier.try_exec("cat", "/logs/verifier/ctrf.json")
        try:
            tests = json.loads(out.stdout)["results"]["tests"]
        except (json.JSONDecodeError, KeyError):
            return RunResult({}, [Problem(Status.NO_RESULTS)], {})
        results = {t["name"]: t["status"] == "passed" for t in tests}
        # the traceback says where it failed, the message often says nothing
        errors = {
            t["name"]: t.get("trace") or t.get("message") or ""
            for t in tests
            if t["status"] != "passed"
        }
        return RunResult(results, [], errors)


# {container path: host path} for .py files under the copied artifacts
def code_files(artifacts: dict) -> dict[str, Path]:
    files = {}
    for artifact, local in artifacts.items():
        for host in local.rglob("*.py"):
            if host.is_file():
                rel = host.relative_to(local).as_posix()
                files[artifact.rstrip("/") + "/" + rel] = host
    return dict(sorted(files.items()))


# copies artifact dirs from the host into the container
def _copy_in(container, artifacts):
    for path, local in artifacts.items():
        container.exec("mkdir", "-p", path)
        container.copy_to(f"{local}/.", path)
