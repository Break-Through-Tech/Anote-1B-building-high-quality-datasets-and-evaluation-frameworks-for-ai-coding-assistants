# `data/`

| Path | Contents |
|------|----------|
| `raw/` | Dataset files exactly as downloaded — not edited by hand |
| `processed/` | Cleaned / formatted data produced by our code |

## Why the data isn't committed

Everything here is gitignored except this file and the `.gitkeep` placeholders. The dataset
is downloadable from a versioned public source, so a download command is more reliable than
a copy in Git — and large files committed to Git stay in history permanently, even after
deletion.

## Getting the data

```bash
pip install huggingface_hub
huggingface-cli download harborframework/terminal-bench-2.0 \
  --repo-type dataset --local-dir data/raw/terminal-bench-2.0
```

Source of truth: <https://github.com/harbor-framework/terminal-bench-2-1/tree/main/tasks>
