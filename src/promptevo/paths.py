"""Where the repo keeps its code, data and results."""

from pathlib import Path

ROOT    = Path(__file__).parents[2]
SRC     = ROOT / "src"
DATA    = ROOT / "data"     # downloads, gitignored
RESULTS = ROOT / "results"  # run logs and caches, gitignored
