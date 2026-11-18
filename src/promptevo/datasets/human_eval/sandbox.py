"""Runs model code and its asserts inside Docker instead of on our machine.

Every job in a batch runs in one python:3.12-slim container with no network,
8 at a time. Each assert gets 3 s, all of a job's asserts get 10 s together,
and a job that still hangs (e.g. stuck in C code) is killed at 20 s.

With a ResultCache, a job that already ran somewhere gets its stored result
instead. The sandbox gives the same result for the same job every time, so
this only saves time.
"""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple

from promptevo.llm.cache import SqliteCache

IMAGE = "python:3.12-slim"


# json turns these into lists, which is what the harness reads and what the
# cache keys were made from
class Job(NamedTuple):
    assertions:  list[str]
    code:        str
    entry_point: str  # the function the asserts call as candidate


class JobResult(NamedTuple):
    passing: dict[str, bool]  # {assertion: passed?}
    error:   str | None       # exception name if the code itself failed

# HumanEval.passing_cases plus time limits. prints {"passing", "error"}.
# the asserts share 10 s like bigcode's humanevalfix harness, so a fix that
# loops can't hold up the batch for 3 s per assert. asserts left when it
# runs out fail
_HARNESS = '''\
import json, signal, sys, time

def _timeout(*_):
    raise TimeoutError()

def _limit(seconds):
    signal.setitimer(signal.ITIMER_REAL, seconds)

signal.signal(signal.SIGALRM, _timeout)
assertions, code, entry_point = json.load(open(sys.argv[1]))
passing = {assertion: False for assertion in assertions}
error = None
try:
    namespace = {}
    _limit(5)
    exec(code, namespace)
    _limit(0)
    namespace["candidate"] = namespace[entry_point]
    deadline = time.monotonic() + 10
    for assertion in assertions:
        left = deadline - time.monotonic()
        if left <= 0:
            break
        try:
            _limit(min(3, left))
            exec(assertion, namespace)
            passing[assertion] = True
        except Exception:
            passing[assertion] = False
        finally:
            _limit(0)
except BaseException as e:
    error = type(e).__name__
print()
print(json.dumps({"passing": passing, "error": error}))
'''

# runs the harness on every job with a time limit, 8 at a time, prints
# {job: output}
_DRIVER = '''\
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

def run(name):
    try:
        p = subprocess.run(
            [sys.executable, "/work/harness.py", f"/work/{name}.json"],
            capture_output=True, text=True, timeout=20,
        )
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"passing": None, "error": type(e).__name__}

names = json.load(open("/work/jobs.json"))
with ThreadPoolExecutor(8) as pool:
    print(json.dumps(dict(zip(names, pool.map(run, names)))))
'''


class ResultCache:
    """sqlite file of job results that runs share. the key covers the
    harness too, so changing how jobs run never reuses an old result. it
    only has the image's tag though, start a new file if the image
    changes."""

    def __init__(self, path: str | Path):
        self._store = SqliteCache(path, table="results", column="result")

    def get(self, key: str) -> JobResult | None:
        """the stored result, None if this job never ran."""
        text = self._store.get(key)
        return None if text is None else JobResult(*json.loads(text))

    def put(self, results: dict[str, JobResult]):
        """stores {key: result}."""
        self._store.put({key: json.dumps(r) for key, r in results.items()})


def run_jobs(
    jobs: dict, cache: ResultCache | None = None
) -> dict[object, JobResult]:
    """runs {key: Job} and returns {key: JobResult}. with a cache, jobs it
    has don't run again and identical jobs in the batch run once."""
    if cache is None:
        return _run(jobs)[0]
    hashes = {key: _hash(job) for key, job in jobs.items()}
    results, todo = {}, {}
    for key, job in jobs.items():
        hit = cache.get(hashes[key])
        if hit is not None:
            results[key] = hit
        else:
            todo.setdefault(hashes[key], job)
    fresh, unfinished = _run(todo) if todo else ({}, set())
    # a job that died as a whole (hung in C code, docker trouble) can depend
    # on load, so it runs again next time
    cache.put({h: r for h, r in fresh.items() if h not in unfinished})
    for key in jobs:
        if key not in results:
            results[key] = fresh[hashes[key]]
    return results


def _hash(job):
    text = json.dumps([_HARNESS, _DRIVER, IMAGE, *job])
    return hashlib.sha256(text.encode()).hexdigest()


# runs the jobs in one container. returns {key: JobResult} and the keys of
# jobs that died as a whole
def _run(jobs):
    with tempfile.TemporaryDirectory() as tmp:
        names = {}
        for i, (key, job) in enumerate(jobs.items()):
            names[key] = f"job{i}"
            Path(tmp, f"job{i}.json").write_text(json.dumps(job))
        Path(tmp, "jobs.json").write_text(json.dumps(list(names.values())))
        Path(tmp, "harness.py").write_text(_HARNESS)
        Path(tmp, "driver.py").write_text(_DRIVER)
        out = subprocess.run(
            [
                "docker", "run", "--rm", "--network", "none",
                "--memory", "1g", "-v", f"{tmp}:/work:ro",
                IMAGE, "python", "/work/driver.py",
            ],
            capture_output=True,
            text=True,
            timeout=60 + 25 * len(jobs),
            check=True,
        )
    raw = json.loads(out.stdout)
    results, unfinished = {}, set()
    for key, name in names.items():
        if raw[name]["passing"] is None:
            unfinished.add(key)
        # a job that timed out as a whole fails every assert
        passing = raw[name]["passing"] or dict.fromkeys(
            jobs[key].assertions, False
        )
        results[key] = JobResult(passing, raw[name]["error"])
    return results, unfinished
