"""The validation pick and the test-set comparison, shared by the CLI, the
transfer runs and notebooks."""

from enum import StrEnum

from promptevo.scoring.evaluation import Evaluation, Evaluator
from promptevo.scoring.individual import Individual


# names on the test set, next to one "seed <i>" per seed
class Entry(StrEnum):
    EVOLVED   = "evolved"    # best train score
    VAL_PICK  = "val pick"   # best validation score
    BEST_SEED = "best seed"
    NO_PROMPT = "no prompt"


def seed_entry(i: int) -> str:
    """the test set's name for seed i."""
    return f"seed {i}"


def is_seed_entry(name: str) -> bool:
    return name.startswith("seed ")


def pick_by_validation(
    validator: Evaluator, prompts: list[str]
) -> tuple[str, dict[str, float]]:
    """scores each prompt on the validation set. returns the best one and
    {prompt: validation score}."""
    scores = {}
    for prompt in prompts:
        if prompt not in scores:
            scores[prompt] = validator.evaluate(prompt).score
    # ties go to the earlier prompt
    return max(scores, key=scores.get), scores


def test_prompts(
    best: Individual,
    seeds: list[Individual],
    picked: str | None = None,
    validation: dict[str, float] | None = None,
) -> dict[str, str]:
    """{name: prompt} to score on the test set. evolved vs the best seed
    shows what the generations added on top of picking a seed. with
    validation ({prompt: score}) the best seed is picked by it, so every
    method's baseline is chosen the same way."""
    # ties go to the earlier seed
    if validation is not None:
        best_seed = max(seeds, key=lambda ind: validation[ind.prompt])
    else:
        best_seed = max(seeds, key=lambda ind: ind.evaluation.score)
    prompts = {Entry.EVOLVED: best.prompt}
    if picked is not None:
        prompts[Entry.VAL_PICK] = picked
    prompts |= {
        Entry.BEST_SEED: best_seed.prompt,
        # empty system message, the chat template then adds no default one
        Entry.NO_PROMPT: "",
    }
    for i, seed in enumerate(seeds):
        prompts[seed_entry(i)] = seed.prompt
    # plain strings, so a printed dict shows the names and not enum members
    return {str(name): prompt for name, prompt in prompts.items()}


def score_test_set(
    tester: Evaluator, prompts: dict[str, str]
) -> dict[str, Evaluation]:
    """{name: evaluation on the test set}. the evaluator caches, so a prompt
    under two names only runs once."""
    return {name: tester.evaluate(prompt) for name, prompt in prompts.items()}
