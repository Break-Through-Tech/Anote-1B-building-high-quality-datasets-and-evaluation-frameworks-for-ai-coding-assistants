"""Thin wrapper over the Docker CLI."""

import subprocess
from pathlib import Path


# runs a docker command and returns its stdout
def docker(*args) -> str:
    out = subprocess.run(
        ["docker", *args], check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


def image_exists(image: str) -> bool:
    found = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True
    )
    return found.returncode == 0


def remove_images(*images: str):
    subprocess.run(["docker", "rmi", "-f", *images], capture_output=True)


class Container:
    """`with Container(image) as c:` starts it idle, removes it after."""

    def __init__(self, image: str):
        self.image = image
        self.id    = None

    def __enter__(self):
        self.id = docker("run", "-d", self.image, "sleep", "infinity")
        return self

    def __exit__(self, *exc):
        subprocess.run(["docker", "rm", "-f", self.id], capture_output=True)

    # raises if the command fails
    def exec(self, *cmd) -> str:
        return docker("exec", self.id, *cmd)

    # doesn't raise on failure. raw bytes, test output can be anything
    def try_exec(self, *cmd, timeout=None) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["docker", "exec", self.id, *cmd],
            capture_output=True,
            timeout=timeout,
        )

    def copy_to(self, host: str | Path, path: str):
        docker("cp", host, f"{self.id}:{path}")

    def copy_from(self, path: str, host: str | Path):
        docker("cp", f"{self.id}:{path}", host)
