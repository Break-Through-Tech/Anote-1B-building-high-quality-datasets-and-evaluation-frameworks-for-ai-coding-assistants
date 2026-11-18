"""Mutation operators from Promptbreeder and GEPA.

Each Mutation turns a parent prompt into a child with the mutator model.
MixedOperators picks one at random for every child, like Promptbreeder does
(sec 3.2). The text they send is in templates.py.

Promptbreeder: https://arxiv.org/abs/2309.16797
GEPA: https://arxiv.org/abs/2507.19457
"""

import difflib
import random
import re
from abc import ABC, abstractmethod

from promptevo.llm.models import ChatModel
from promptevo.scoring.evaluation import Evaluation
from promptevo.scoring.individual import Individual
from promptevo.search.feedback import Feedback, item_feedback
from promptevo.search.operators import LLMOperators
from promptevo.search.runlog import RunLog
from promptevo.search.templates import (
    APPLY_MUTATION_PROMPT,
    EDA_LIST,
    EDA_RANK,
    EXAMPLE,
    HINTS,
    IMPROVE_MUTATION_PROMPT,
    LAMARCKIAN,
    LINEAGE,
    MUTATION_PROMPTS,
    NEW_MUTATION_PROMPT,
    REFLECT,
    REFLECT_EXAMPLE,
    ROLE,
    THINKING_STYLES,
)


class Mutation(ABC):
    """one way to make a child prompt from a parent."""

    name: str

    @abstractmethod
    def apply(
        self, ops: "MixedOperators", parent: str, evaluation: Evaluation
    ) -> str:
        ...


class Rewrite(Mutation):
    """one change to the prompt, with operators.py's template."""

    name = "rewrite"

    def apply(self, ops, parent, evaluation):
        return ops.rewrite(parent, evaluation)


class ZeroOrder(Mutation):
    """promptbreeder zero-order prompt generation. a new prompt from the task
    description alone, the parent isn't used."""

    name = "zero-order"

    def apply(self, ops, parent, evaluation):
        return _first_item(ops.ask(HINTS.format(description=ops.description)))


class FirstOrder(Mutation):
    """promptbreeder first-order prompt generation. the parent's mutation
    prompt applied to the parent."""

    name = "first-order"

    def apply(self, ops, parent, evaluation):
        return ops.apply_mutation_prompt(ops.mutation_prompt(parent), parent)


class EDA(Mutation):
    """promptbreeder eda mutation. continue a list of the population's
    prompts, near-duplicates filtered out, no scores shown."""

    name = "eda"

    def apply(self, ops, parent, evaluation):
        prompts = _distinct([ind.prompt for ind in ops.population])
        ops.rng.shuffle(prompts)
        return ops.ask(EDA_LIST.format(prompts=_numbered(prompts)))


class EDARank(Mutation):
    """promptbreeder eda rank and index mutation. the list is sorted worst
    to best but labelled best to worst, on purpose (sec 3.2.2)."""

    name = "eda-rank"

    def apply(self, ops, parent, evaluation):
        ranked = sorted(ops.population, key=lambda ind: ind.evaluation.score)
        prompts = _distinct([ind.prompt for ind in ranked])
        return ops.ask(EDA_RANK.format(
            mutation_prompt=ops.mutation_prompt(parent),
            best=len(prompts) + 1,
            n=len(prompts),
            prompts=_numbered(prompts),
        ))


class Lineage(Mutation):
    """promptbreeder lineage based mutation. continue the history of best
    prompts, oldest first."""

    name = "lineage"

    def apply(self, ops, parent, evaluation):
        return ops.ask(LINEAGE.format(prompts=_numbered(ops.elites)))


class ZeroOrderHyper(Mutation):
    """promptbreeder zero-order hyper-mutation. a new mutation prompt from
    the task description and a thinking style, then applied to the
    parent. the child keeps the new mutation prompt."""

    name = "zero-order-hyper"

    def apply(self, ops, parent, evaluation):
        style = ops.rng.choice(THINKING_STYLES)
        mutation_prompt = ops.ask(NEW_MUTATION_PROMPT.format(
            description=ops.description, style=style
        ))
        child = ops.apply_mutation_prompt(mutation_prompt, parent)
        ops.remember(child, mutation_prompt)
        return child


class FirstOrderHyper(Mutation):
    """promptbreeder first-order hyper-mutation. improves the parent's
    mutation prompt, then applies it. the child keeps the new one."""

    name = "first-order-hyper"

    def apply(self, ops, parent, evaluation):
        mutation_prompt = ops.ask(IMPROVE_MUTATION_PROMPT.format(
            mutation_prompt=ops.mutation_prompt(parent)
        ))
        child = ops.apply_mutation_prompt(mutation_prompt, parent)
        ops.remember(child, mutation_prompt)
        return child


class Lamarckian(Mutation):
    """promptbreeder working out to task-prompt. guesses the instruction
    from one item the parent got fully right. rewrites instead if there
    isn't one."""

    name = "lamarckian"

    def apply(self, ops, parent, evaluation):
        solved = [
            item for item, score in evaluation.items.items()
            if score == 1.0 and evaluation.results[item].output
        ]
        if not solved:
            return ops.rewrite(parent, evaluation)
        result = evaluation.results[ops.rng.choice(sorted(solved))]
        example = _example(result.inputs, result.output)
        return ops.ask(LAMARCKIAN.format(example=example))


class Reflective(Mutation):
    """gepa reflective prompt mutation. shows the mutator 3 of the parent's
    attempts with feedback and asks for a better instruction."""

    name = "reflective"
    minibatch = 3  # gepa app. E.4

    def apply(self, ops, parent, evaluation):
        results = evaluation.results
        items = ops.rng.sample(
            sorted(results), min(self.minibatch, len(results))
        )
        examples = "\n\n".join(
            REFLECT_EXAMPLE.format(
                example=_example(results[i].inputs, results[i].output),
                feedback=item_feedback(results[i], ops.show_tests),
            )
            for i in items
        )
        return ops.ask(REFLECT.format(prompt=parent, examples=examples))


MUTATIONS = {
    m.name: m
    for m in [
        Rewrite, ZeroOrder, FirstOrder, EDA, EDARank, Lineage,
        ZeroOrderHyper, FirstOrderHyper, Lamarckian, Reflective,
    ]
}


class MixedOperators(LLMOperators):
    """llm operators where every mutation is one of several, picked at
    random. also keeps what promptbreeder's operators need, the current
    population, the best prompt of every generation and a mutation prompt
    per prompt."""

    def __init__(
        self,
        model: ChatModel,
        mutations: list[Mutation],
        description: str,
        feedback: Feedback = Feedback.NONE,
        seed: int = 0,
        log: RunLog | None = None,
        show_tests: bool = True,
    ):
        super().__init__(model, feedback, seed)
        self.mutations   = mutations
        self.description = description
        self.log         = log or RunLog()
        # False hides the failing asserts (and their expected values) from
        # reflective mutation
        self.show_tests  = show_tests
        # which mutation, which items, which thinking style. its own stream,
        # so these choices don't shift the per-call seeds
        self.rng = random.Random(seed + 1)
        self.population: list[Individual] = []
        self.elites: list[str] = []
        self._mutation_prompts = {}  # {prompt: its mutation prompt}

    def observe(self, population):
        self.population = population
        if not self.elites or self.elites[-1] != population[0].prompt:
            self.elites.append(population[0].prompt)

    def mutate(self, prompt, evaluation):
        mutation = self.mutations[0]
        if len(self.mutations) > 1:
            mutation = self.rng.choice(self.mutations)
        child = _clean(mutation.apply(self, prompt, evaluation))
        # code means the mutator solved a task instead of writing a prompt.
        # empty makes the ea keep the parent
        if "```" in child:
            child = ""
        self._inherit(child, prompt)
        self.log.mutation(mutation.name, prompt, child)
        return child

    def crossover(self, a, b):
        child = super().crossover(a, b)
        self._inherit(child, a)
        return child

    def ask(self, text: str) -> str:
        """sends text to the mutator, returns the reply without labels."""
        seed = self._seeds.randrange(2**31)
        return _clean(self.model.chat(text, system=ROLE, seed=seed).strip())

    def apply_mutation_prompt(self, mutation_prompt: str, prompt: str) -> str:
        """promptbreeder's mutation prompt + task prompt format."""
        return self.ask(APPLY_MUTATION_PROMPT.format(
            mutation_prompt=mutation_prompt, prompt=prompt
        ))

    # a random one from MUTATION_PROMPTS the first time a prompt needs one,
    # so runs that never use them draw nothing from the rng
    def mutation_prompt(self, prompt: str) -> str:
        """the mutation prompt that goes with prompt."""
        if prompt not in self._mutation_prompts:
            self._mutation_prompts[prompt] = self.rng.choice(MUTATION_PROMPTS)
        return self._mutation_prompts[prompt]

    def remember(self, prompt: str, mutation_prompt: str):
        """gives prompt its own mutation prompt."""
        self._mutation_prompts[prompt] = mutation_prompt

    # children keep their parent's mutation prompt unless they got a new one
    def _inherit(self, child, parent):
        if child not in self._mutation_prompts and (
            parent in self._mutation_prompts
        ):
            self._mutation_prompts[child] = self._mutation_prompts[parent]


# the first line of a list reply, without its number or bullet
def _first_item(reply):
    for line in reply.splitlines():
        line = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", line).strip()
        # skips a chat preamble like "Here are 100 hints:"
        if line and not line.endswith(":"):
            return line
    return ""


# drops prompts nearly the same as one already kept. promptbreeder used bert
# embeddings over 0.95 cosine, difflib keeps this dependency free
def _distinct(prompts):
    kept = []
    for prompt in prompts:
        if all(
            difflib.SequenceMatcher(None, prompt, k).ratio() <= 0.95
            for k in kept
        ):
            kept.append(prompt)
    return kept


def _numbered(prompts):
    return "\n".join(f"{i}. {p}" for i, p in enumerate(prompts, 1))


# long inputs get cut so 3 examples fit in the mutator's context
def _example(inputs, output, limit=1500):
    return EXAMPLE.format(inputs=inputs[:limit], output=output[:limit])


# labels and list numbers small models like to put in front of the prompt
def _clean(text):
    text = re.sub(
        r"^\s*(INSTRUCTION MUTANT|INSTRUCTION|New prompt|Prompt)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^\s*\d+[.)]\s+", "", text)
    return text.strip().strip('"').strip()
