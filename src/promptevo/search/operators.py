"""Mutate and crossover. What the mutator is told about a parent is in
feedback.py."""

import random
from abc import ABC, abstractmethod

from promptevo.llm.models import ChatModel
from promptevo.scoring.evaluation import Evaluation
from promptevo.scoring.individual import Individual
from promptevo.search.feedback import Feedback, describe
from promptevo.search.templates import CROSSOVER, MUTATE


class Operators(ABC):
    """what a search uses to make new prompts. subclass to try other ones."""

    @abstractmethod
    def mutate(self, prompt: str, evaluation: Evaluation) -> str: ...

    @abstractmethod
    def crossover(self, a: str, b: str) -> str: ...

    def observe(self, population: list[Individual]):
        """sees the survivors after every generation. most operators don't
        care."""

    @property
    def mutator_calls(self) -> int | None:
        """chat calls the mutator model has made, None without one."""
        return None


class LLMOperators(Operators):
    """an llm rewrites/combines prompts."""

    def __init__(
        self,
        model: ChatModel,
        feedback: Feedback = Feedback.NONE,
        seed: int = 0,
    ):
        self.model    = model
        self.feedback = feedback
        # a new seed every call, so the same run always gets the same
        # rewrites but asking twice still gives two different ones
        self._seeds = random.Random(seed)

    @property
    def mutator_calls(self):
        return self.model.calls

    def mutate(self, prompt, evaluation):
        return self.rewrite(prompt, evaluation)

    def rewrite(self, prompt: str, evaluation: Evaluation) -> str:
        """one change to the prompt, with MUTATE."""
        shown = describe(evaluation, self.feedback)
        return self._ask(MUTATE.format(prompt=prompt, feedback=shown))

    def crossover(self, a, b):
        return self._ask(CROSSOVER.format(a=a, b=b))

    # the templates ask for the bare prompt, so the reply is the prompt.
    # small models often ignore a request for a fenced block
    def _ask(self, text):
        reply = self.model.chat(text, seed=self._seeds.randrange(2**31))
        return reply.strip()
