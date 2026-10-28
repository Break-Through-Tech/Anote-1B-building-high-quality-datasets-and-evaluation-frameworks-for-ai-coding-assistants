"""Mutate and crossover. Feedback controls what the mutator sees."""

import random
import re
from abc import ABC, abstractmethod
from enum import StrEnum

from promptevo.models import OllamaModel
from promptevo.types import Evaluation

_MUTATE = """\
You are mutating a system prompt for a coding assistant.

Make EXACTLY one change that will improve the assistant's performance on the
tasks. Preserve the original purpose.
{feedback}
Original prompt:
{prompt}

Return the modified prompt inside a ``` fenced code block."""

_CROSSOVER = """\
You are performing a crossover operation on two system prompts for a coding
assistant. Combine the information from both parent prompts into one new
child prompt that keeps their original intent.

Parent A:
{a}

Parent B:
{b}

Return the new child prompt inside a ``` fenced code block."""

# a fence only counts at the start of a line, so a prompt that mentions
# ```python mid-sentence doesn't end the block early
_FENCE_RE = re.compile(r"^```[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)


class Operators(ABC):
    """what the ea uses to make new prompts. subclass to try other ones."""

    @abstractmethod
    def mutate(self, prompt: str, evaluation: Evaluation) -> str: ...

    @abstractmethod
    def crossover(self, a: str, b: str) -> str: ...


# what the mutator sees
class Feedback(StrEnum):
    NONE     = "none"      # blind rewrite
    SCORE    = "score"     # score only
    FAILURES = "failures"  # score + failing tests


class LLMOperators(Operators):
    """an llm rewrites/combines prompts."""

    def __init__(
        self,
        model: OllamaModel,
        feedback: Feedback = Feedback.NONE,
        seed: int = 0,
    ):
        self.model    = model
        self.feedback = feedback
        # new seed every call, so the same run always gets the same rewrites
        # but asking twice still gives two different ones
        self._rng = random.Random(seed)

    def mutate(self, prompt, evaluation):
        shown = _describe(evaluation, self.feedback)
        return self._ask(_MUTATE.format(prompt=prompt, feedback=shown))

    def crossover(self, a, b):
        return self._ask(_CROSSOVER.format(a=a, b=b))

    # the prompt inside the reply's fence. "" if there isn't one, the ea then
    # keeps the parent
    def _ask(self, text):
        reply = self.model.chat(text, seed=self._rng.randrange(2**31))
        fenced = _FENCE_RE.search(reply)
        return fenced.group(1).strip() if fenced else ""


def _describe(evaluation, feedback):
    if feedback == Feedback.NONE:
        return ""
    text = (
        f"\nWith this prompt the assistant scored {evaluation.score:.2f}"
        " (share of failing tests it fixed, 0 to 1).\n"
    )
    if feedback == Feedback.FAILURES and evaluation.feedback:
        text += f"What happened per task:\n{evaluation.feedback}\n"
    return text
