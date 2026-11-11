"""Runs model code and its asserts inside Docker instead of on our machine.

Every job in a batch runs in one python:3.12-slim container with no network.
Each assert gets 3 s and each job gets 30 s in total.
"""

import json
import subprocess
import tempfile
from pathlib import Path

IMAGE = "python:3.12-slim"

# HumanEval.passing_cases plus time limits. prints {"passing", "error"}
_HARNESS = '''\
import json, signal, sys

def _timeout(*_):
    raise TimeoutError()

signal.signal(signal.SIGALRM, _timeout)
assertions, code, entry_point = json.load(open(sys.argv[1]))
passing = {assertion: False for assertion in assertions}
error = None
try:
    namespace = {}
    signal.alarm(5)
    exec(code, namespace)
    signal.alarm(0)
    namespace["candidate"] = namespace[entry_point]
    for assertion in assertions:
        try:
            signal.alarm(3)
            exec(assertion, namespace)
            passing[assertion] = True
        except Exception:
            passing[assertion] = False
        finally:
            signal.alarm(0)
except BaseException as e:
    error = type(e).__name__
print()
print(json.dumps({"passing": passing, "error": error}))
'''

# runs the harness on every job with a time limit, prints {job: output}
_DRIVER = '''\
import json, subprocess, sys
results = {}
for name in json.load(open("/work/jobs.json")):
    try:
        p = subprocess.run(
            [sys.executable, "/work/harness.py", f"/work/{name}.json"],
            capture_output=True, text=True, timeout=30,
        )
        results[name] = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:
        results[name] = {"passing": None, "error": type(e).__name__}
print(json.dumps(results))
'''


def run_jobs(jobs):
    """{key: (assertions, code, entry_point)} to {key: (passing, error)}.
    passing is {assertion: bool}, error is the exception name or None."""
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
            timeout=60 + 35 * len(jobs),
            check=True,
        )
    raw = json.loads(out.stdout)
    results = {}
    for key, name in names.items():
        assertions = jobs[key][0]
        # a job that timed out as a whole fails every assert
        passing = raw[name]["passing"] or dict.fromkeys(assertions, False)
        results[key] = (passing, raw[name]["error"])
    return results
