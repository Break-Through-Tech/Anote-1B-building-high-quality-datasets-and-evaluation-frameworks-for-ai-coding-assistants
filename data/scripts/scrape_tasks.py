import json
import os
import sys 

try:
    import tomllib
except ImportError:
    import toml as tomllib_fallback
    
TASK_PATH = "/Users/angelaliu/terminal-bench/tasks"
 
# Keywords to match against category AND tags (case-insensitive, substring match)
DEBUG_KEYWORDS = ["debug", "bug-fix", "bugfix", "fix"]
CODEGEN_KEYWORDS = ["code-generation", "codegen", "generation", "programming", "coding"]
 
OUTPUT_DIR = "matching_tasks.json"


def load_toml(path):
    if "tomllib" in sys.modules:
        with open(path, "rb") as f:
            return tomllib.load(f)
    else:
        with open(path) as f:
            return tomllib_fallback.load(f)
 
 
def matches_keywords(category, tags, keywords):
    text = (category or "").lower() + " " + " ".join(t.lower() for t in tags)
    return any(kw in text for kw in keywords)
 
 
def main():
    debug_matches = []
    codegen_matches = []
    unmatched = []
 
    for task_name in sorted(os.listdir(TASK_PATH)):
        task_path = os.path.join(TASK_PATH, task_name)
        toml_path = os.path.join(task_path, "task.toml")
        if not os.path.isdir(task_path) or not os.path.exists(toml_path):
            continue
 
        try:
            metadata = load_toml(toml_path).get("metadata", {})
        except Exception as e:
            print(f"couldn't parse {toml_path}: {e}")
            continue
 
        category = metadata.get("category", "")
        tags = metadata.get("tags", [])
 
        is_debug = matches_keywords(category, tags, DEBUG_KEYWORDS)
        is_codegen = matches_keywords(category, tags, CODEGEN_KEYWORDS)
 
        if is_debug:
            debug_matches.append(task_name)
        if is_codegen:
            codegen_matches.append(task_name)
        if not is_debug and not is_codegen:
            unmatched.append({"task": task_name, "category": category, "tags": tags})
 
    print(f"Debugging matches: {len(debug_matches)}")
    for t in debug_matches:
        print(f"  {t}")
 
    print(f"\nCode-generation matches: {len(codegen_matches)}")
    for t in codegen_matches:
        print(f"  {t}")
 
    print(f"\nUnmatched tasks: {len(unmatched)} (review these manually - some might still fit)")
 
    with open(OUTPUT_DIR, "w") as f:
        json.dump(
            {
                "debugging": debug_matches,
                "code_generation": codegen_matches,
                "unmatched": unmatched,
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {OUTPUT_DIR}")
 
 
if __name__ == "__main__":
    main()
 