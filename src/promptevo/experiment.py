"""Seed prompts and the test-set comparison, shared by the CLI and notebooks."""

from promptevo.evaluation import Evaluator
from promptevo.types import Individual

SEED_PROMPTS = [
    "You are an expert level software engineer. Solve the coding task "
    "carefully, following all directions.",
    "You are a coding assistant. Make sure you thoroughly analyze the "
    "question before solving.",
    "You are an experienced programmer. Make the least amount of changes to "
    "produce a more efficient solution, while preserving functionality.",
    "You are a meticulous software engineer. Consider edge cases, and make "
    "sure your solution is correct.",
]


def test_prompts(
    best: Individual, seeds: list[Individual]
) -> dict[str, str]:
    """{name: prompt} to score on the test set. evolved vs the best seed
    shows what the generations added on top of picking a seed."""
    # highest train score, ties go to the earlier seed
    best_seed = max(seeds, key=lambda ind: ind.evaluation.score)
    prompts = {
        "evolved": best.prompt,
        "best seed": best_seed.prompt,
        # empty system message, ollama then adds no default one either
        "no prompt": "",
    }
    for i, seed in enumerate(seeds):
        prompts[f"seed {i}"] = seed.prompt
    return prompts


def score_test_set(
    tester: Evaluator, prompts: dict[str, str]
) -> dict[str, float]:
    """{name: test score}. each distinct prompt is only scored once."""
    by_prompt = {}
    for prompt in prompts.values():
        if prompt not in by_prompt:
            by_prompt[prompt] = tester.evaluate(prompt).score
    return {name: by_prompt[prompt] for name, prompt in prompts.items()}
