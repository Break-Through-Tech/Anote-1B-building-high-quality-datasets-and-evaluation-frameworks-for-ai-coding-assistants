"""reads each terminal-bench task folder and saves it as one clean record. output file has one task per line"""

import argparse
import json
import os
import re
import sys
import tomllib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TASKS_DIR = os.path.join(REPO_ROOT, "data", "raw", "terminal-bench", "tasks")
DEFAULT_OUT = os.path.join(REPO_ROOT, "data", "processed", "tasks.jsonl")

# file types we count as code, and what language each one is. anything else (images, pdfs, data files) is ignored
CODE_EXTS = {
    ".py": "python", ".sh": "bash", ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp", ".cu": "cuda",
    ".rs": "rust", ".ts": "typescript", ".tsx": "typescript", ".js": "javascript", ".jsx": "javascript",
    ".scala": "scala", ".v": "verilog_or_coq", ".lean": "lean", ".sage": "sage", ".rq": "sparql",
    ".patch": "diff", ".diff": "diff",
}

# every instruction starts with a hidden "do not train on this" marker. this finds it so we can take it out
CANARY_RE = re.compile(r"<!--[^>]*harbor-canary[^>]*-->\s*", re.IGNORECASE)


def difficulty_from_hours(hours):
    # terminal-bench v4 doesn't label difficulty, but it does say how many hours an expert would need.
    # so we turn hours into easy / medium / hard.
    if hours is None:
        return None
    if hours <= 2:
        return "easy"
    if hours <= 6:
        return "medium"
    return "hard"


def long_path(path):
    # lets windows open paths over 260 chars (some tasks nest deep)
    if os.name == "nt":
        return "\\\\?\\" + os.path.abspath(path)
    return path


def read_code_files(folder):
    # reads every code file in a folder (and its subfolders) into {path: file contents}
    code = {}
    folder = long_path(folder)
    for root, _, files in os.walk(folder):
        for name in files:
            if os.path.splitext(name)[1].lower() in CODE_EXTS:
                path = os.path.join(root, name)
                rel = os.path.relpath(path, folder).replace(os.sep, "/")
                with open(path, encoding="utf-8", errors="replace") as f:
                    code[rel] = f.read()
    return dict(sorted(code.items()))


def mentioned_in(script, rel):
    # true if solve.sh names this file or a folder it's in
    parts = rel.split("/")
    names = [rel, parts[-1]] + ["/".join(parts[:i]) for i in range(1, len(parts))]
    return any(re.search(r"(?<![\w.-])" + re.escape(n) + r"(?![\w.-])", script) for n in names)


def answer_files(solution_folder):
    # the answer is whatever solve.sh runs or copies, since some folders have extra helper files.
    # falls back to every code file if solve.sh names none
    files = read_code_files(solution_folder)
    script = files.pop("solve.sh", "")
    answer = {rel: text for rel, text in files.items() if mentioned_in(script, rel)}
    return answer or files


def main_language(files):
    # the language with the most code across these files
    sizes = {}
    for rel, text in files.items():
        lang = CODE_EXTS[os.path.splitext(rel)[1].lower()]
        sizes[lang] = sizes.get(lang, 0) + len(text)
    return max(sizes, key=sizes.get) if sizes else None


def build_record(task_dir):
    with open(os.path.join(task_dir, "task.toml"), "rb") as f:
        meta = tomllib.load(f).get("metadata", {})
    with open(os.path.join(task_dir, "instruction.md"), encoding="utf-8") as f:
        instruction = CANARY_RE.sub("", f.read()).strip()

    solution = answer_files(os.path.join(task_dir, "solution"))
    tests = read_code_files(os.path.join(task_dir, "tests"))
    tests.pop("test.sh", None)  # just a launcher that runs the tests

    author = meta.get("author_name", "")
    if isinstance(author, list):  # a few tasks have more than one author
        author = ", ".join(author)

    return {
        "task_id": os.path.basename(task_dir),
        "instruction": instruction,
        "metadata": {
            "author_name": author,
            "difficulty": difficulty_from_hours(meta.get("expert_time_estimate_hours")),
            "category": meta.get("category", ""),
            "subcategory": meta.get("subcategory", ""),
            "tags": meta.get("tags", []),
            "expert_time_estimate_hours": meta.get("expert_time_estimate_hours"),
        },
        "solution_language": main_language(solution),
        "solution": solution,
        "tests": tests,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", default=DEFAULT_TASKS_DIR, help="terminal-bench tasks/ folder")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output .jsonl path")
    parser.add_argument("--tasks", nargs="*", help="only these task names (default: all)")
    parser.add_argument("--language", help="only tasks whose answer is mostly this language, e.g. python")
    args = parser.parse_args()

    if not os.path.isdir(args.tasks_dir):
        sys.exit(f"no tasks folder at {args.tasks_dir}, see data/README.md for how to download it")

    names = sorted(n for n in os.listdir(args.tasks_dir) if os.path.isfile(os.path.join(args.tasks_dir, n, "task.toml")))
    if args.tasks:
        names = [n for n in names if n in args.tasks]

    records = []
    for name in names:
        record = build_record(os.path.join(args.tasks_dir, name))
        if args.language is None or record["solution_language"] == args.language:
            records.append(record)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
