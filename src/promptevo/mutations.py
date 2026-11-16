"""Mutation operators from Promptbreeder and GEPA.

Each Mutation turns a parent prompt into a child with the mutator model.
MixedOperators picks one at random for every child, like Promptbreeder does
(sec 3.2). Templates are the papers' own text where they give it, plus a line
asking for the bare prompt, since a chat model otherwise adds explanations.

Promptbreeder: https://arxiv.org/abs/2309.16797
GEPA: https://arxiv.org/abs/2507.19457
"""

import difflib
import random
import re
from abc import ABC, abstractmethod

from promptevo.evaluation import item_feedback
from promptevo.models import OllamaModel
from promptevo.operators import Feedback, LLMOperators
from promptevo.types import Evaluation, Individual

BARE = (
    "Return only the new prompt. Do not include explanations, formatting "
    "instructions, or Markdown code fences."
)

# the papers' templates read like text to continue. a chat model otherwise
# answers them as the coding assistant ("Sure! Please provide the code")
ROLE = (
    "You write and edit system prompts for an AI coding assistant. Never "
    "solve a coding task yourself and never ask for more input. Reply with "
    "exactly the text that was asked for."
)

# promptbreeder app. C, a subset, word for word
MUTATION_PROMPTS = [
    "Modify the following instruction creatively, giving some advice on how "
    "to solve it:",
    "How would you help an LLM to follow the instruction?",
    "Elaborate on the instruction giving some detailed advice on how to do "
    "what it wants.",
    "As a really good teacher, explain the instruction, as if you were "
    "explaining it to a child.",
    "Imagine you need to follow this instruction. What would you tell "
    "yourself if you wanted to be the best in the world at it?",
    "Rephrase the instruction without using any of the same words. Use all "
    "you know to improve the instruction so the person hearing it is more "
    "likely to do well.",
    "Say that instruction again in another way. DON'T use any of the words "
    "in the original instruction or you're fired.",
    "Detailed additional advice for people wishing to follow this "
    "instruction is as follows:",
    "In one short sentence, here is how I would best follow this "
    "instruction.",
    "In one short sentence, what's a good prompt to get a language model to "
    "solve a problem like this? Notice how I don't use any of the same "
    "words as in the INSTRUCTION.",
    "Promote breaking down problems: Instead of asking the user to solve the "
    "problem as a whole, prompt them to break it down into smaller, more "
    "manageable parts.",
    "Ask for comprehension: Modify the prompt to ask the user to review and "
    "confirm their understanding of all aspects of the problem.",
    "Look carefully to see what you did wrong, how could you fix the "
    "problem?",
]

# promptbreeder app. D, a subset, word for word
THINKING_STYLES = [
    "How could I devise an experiment to help solve that problem?",
    "Make a list of ideas for solving this problem, and apply them one by "
    "one to the problem to see if any progress can be made.",
    "How could I measure progress on this problem?",
    "How can I simplify the problem so that it is easier to solve?",
    "What are the key assumptions underlying this problem?",
    "What are the potential risks and drawbacks of each solution?",
    "What are the alternative perspectives or viewpoints on this problem?",
    "How can I break down this problem into smaller, more manageable parts?",
    "What is the core issue or problem that needs to be addressed?",
    "What are the underlying causes or factors contributing to the problem?",
    "What are the potential obstacles or challenges that might arise in "
    "solving this problem?",
    "What kinds of solution typically are produced for this kind of problem "
    "specification?",
    "Let's imagine the current best solution is totally wrong, what other "
    "ways are there to think about the problem specification?",
    "What is the best way to modify this current best solution, given what "
    "you know about these kinds of problem specification?",
    "Let's think step by step.",
    "Let's make a step by step plan and implement it with good notion and "
    "explanation.",
]

# gepa app. C, word for word except the last line (bare prompt instead of
# a ``` block, small models often skip the fence)
REFLECT = (
    "I provided an assistant with the following instructions to perform a "
    "task for me:\n\n"
    "```\n{prompt}\n```\n\n"
    "The following are examples of different task inputs provided to the "
    "assistant along with the assistant's response for each of them, and "
    "some feedback on how the assistant's response could be better:\n\n"
    "```\n{examples}\n```\n\n"
    "Your task is to write a new instruction for the assistant.\n\n"
    "Read the inputs carefully and identify the input format and infer "
    "detailed task description about the task I wish to solve with the "
    "assistant.\n\n"
    "Read all the assistant responses and the corresponding feedback. "
    "Identify all niche and domain specific factual information about the "
    "task and include it in the instruction, as a lot of it may not be "
    "available to the assistant in the future. The assistant may have "
    "utilized a generalizable strategy to solve the task, if so, include "
    "that in the instruction as well.\n\n"
    + BARE
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
        return LLMOperators.mutate(ops, parent, evaluation)


class ZeroOrder(Mutation):
    """promptbreeder zero-order prompt generation. a new prompt from the task
    description alone, the parent isn't used."""

    name = "zero-order"

    def apply(self, ops, parent, evaluation):
        # the paper's "<description> A list of 100 hints:" made the chat
        # model ask for the code instead
        reply = ops.ask(
            f"Task: {ops.description}\n\nWrite a list of 100 short hints "
            "that would help an AI assistant do this task well."
        )
        return _first_item(reply)


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
        return ops.ask(
            "Continue this list with one new prompt.\n\n"
            + _numbered(prompts) + "\n\n" + BARE
        )


class EDARank(Mutation):
    """promptbreeder eda rank and index mutation. the list is sorted worst
    to best but labelled best to worst, on purpose (sec 3.2.2)."""

    name = "eda-rank"

    def apply(self, ops, parent, evaluation):
        ranked = sorted(ops.population, key=lambda ind: ind.evaluation.score)
        prompts = _distinct([ind.prompt for ind in ranked])
        n = len(prompts)
        header = (
            "INSTRUCTION: " + ops.mutation_prompt(parent)
            + "\n A List of Responses in descending order of score."
            + f" {n + 1} is the best response. It resembles {n} more than"
            + " it does (1)"
        )
        return ops.ask(header + "\n\n" + _numbered(prompts) + "\n\n" + BARE)


class Lineage(Mutation):
    """promptbreeder lineage based mutation. continue the history of best
    prompts, oldest first."""

    name = "lineage"

    def apply(self, ops, parent, evaluation):
        return ops.ask(
            "GENOTYPES FOUND IN ASCENDING ORDER OF QUALITY\n\n"
            + _numbered(ops.elites) + "\n\n" + BARE
        )


class ZeroOrderHyper(Mutation):
    """promptbreeder zero-order hyper-mutation. a new mutation prompt from
    the task description and a thinking style, then applied to the
    parent. the child keeps the new mutation prompt."""

    name = "zero-order-hyper"

    def apply(self, ops, parent, evaluation):
        style = ops.rng.choice(THINKING_STYLES)
        mutation_prompt = ops.ask(
            f"{ops.description} {style}\n\nWrite one instruction for how to "
            "improve a prompt for this task. Return only the instruction."
        )
        child = ops.apply_mutation_prompt(mutation_prompt, parent)
        ops.remember(child, mutation_prompt)
        return child


class FirstOrderHyper(Mutation):
    """promptbreeder first-order hyper-mutation. improves the parent's
    mutation prompt, then applies it. the child keeps the new one."""

    name = "first-order-hyper"

    def apply(self, ops, parent, evaluation):
        mutation_prompt = ops.ask(
            "Please summarize and improve the following instruction: "
            + ops.mutation_prompt(parent)
            + "\n\nReturn only the improved instruction."
        )
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
            item for item, score in (evaluation.items or {}).items()
            if score == 1.0 and evaluation.results[item].output
        ]
        if not solved:
            return Rewrite().apply(ops, parent, evaluation)
        result = evaluation.results[ops.rng.choice(sorted(solved))]
        return ops.ask(
            "I gave a friend an instruction and some advice. Here are the "
            "correct examples of his workings out "
            + _example(result.inputs, result.output)
            + " The instruction was:\n\nDon't repeat the workings out. Write "
            "the general instruction the friend was given, in one or two "
            "sentences. " + BARE
        )


class Reflective(Mutation):
    """gepa reflective prompt mutation. shows the mutator 3 of the parent's
    attempts with feedback and asks for a better instruction."""

    name = "reflective"
    minibatch = 3  # gepa app. E.4

    def apply(self, ops, parent, evaluation):
        results = evaluation.results or {}
        items = ops.rng.sample(
            sorted(results), min(self.minibatch, len(results))
        )
        examples = "\n\n".join(
            _example(results[i].inputs, results[i].output)
            + "\nFeedback: " + item_feedback(results[i], ops.show_tests)
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
        model: OllamaModel,
        mutations: list[Mutation],
        description: str,
        feedback: Feedback = Feedback.NONE,
        seed: int = 0,
        log=None,
        show_tests: bool = True,
    ):
        super().__init__(model, feedback, seed)
        # False hides the failing asserts (and their expected values) from
        # reflective mutation
        self.show_tests = show_tests
        self.mutations   = mutations
        self.description = description
        self.log         = log
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
        if self.log:
            self.log.mutation(mutation.name, prompt, child)
        return child

    def crossover(self, a, b):
        child = super().crossover(a, b)
        self._inherit(child, a)
        return child

    def ask(self, text: str) -> str:
        """sends text to the mutator, returns the reply without labels."""
        seed = self._rng.randrange(2**31)
        return _clean(self.model.chat(text, system=ROLE, seed=seed).strip())

    def apply_mutation_prompt(self, mutation_prompt: str, prompt: str) -> str:
        """promptbreeder's mutation prompt + task prompt format."""
        return self.ask(
            f"{mutation_prompt}  INSTRUCTION: {prompt}  INSTRUCTION MUTANT:"
            f"\n\n{BARE}"
        )

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
    return f"Input:\n{inputs[:limit]}\n\nResponse:\n{output[:limit]}"


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
