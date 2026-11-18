"""Text the models get: the seed prompts, and every template the mutator is
sent. The study's replies are cached by request, so editing any of this
changes results (tests/test_text.py checks it stays the same).

Templates are the papers' own text where they give it, plus a line asking
for the bare prompt, since a chat model otherwise adds explanations.

Promptbreeder: https://arxiv.org/abs/2309.16797
GEPA: https://arxiv.org/abs/2507.19457
"""

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

MUTATE = """\
You are mutating a system prompt for a coding assistant.

Make EXACTLY one change that will improve the assistant's performance on the
tasks. Preserve the original purpose.
{feedback}
Original prompt:
{prompt}

Return only the modified system prompt. Do not include explanations,
formatting instructions, or Markdown code fences."""

# what MUTATE's {feedback} shows, by Feedback
SCORE_SHOWN = (
    "\nWith this prompt the assistant scored {score:.2f}"
    " (share of failing tests it fixed, 0 to 1).\n"
)
FAILURES_SHOWN = "What happened per task:\n{feedback}\n"

CROSSOVER = (
    "You are performing a crossover operation on two system prompts for a "
    "coding\nassistant. Combine the information from both parent prompts "
    "into one new\nchild prompt that keeps their original intent.\n\n"
    "Parent A:\n{a}\n\nParent B:\n{b}\n\n"
    "Return only the text of the new child prompt, not JSON or a diff. Do "
    "not include explanations,\nformatting instructions, or Markdown code "
    "fences."
)

BARE = (
    "Return only the new prompt. Do not include explanations, formatting "
    "instructions, or Markdown code fences."
)

# system prompt for every promptbreeder and gepa operator. the papers'
# templates read like text to continue, so a chat model otherwise answers
# them as the coding assistant ("Sure! Please provide the code")
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

# promptbreeder's mutation prompt + task prompt format
APPLY_MUTATION_PROMPT = (
    "{mutation_prompt}  INSTRUCTION: {prompt}  INSTRUCTION MUTANT:\n\n" + BARE
)

# zero-order. the paper's "<description> A list of 100 hints:" made the chat
# model ask for the code instead
HINTS = (
    "Task: {description}\n\nWrite a list of 100 short hints that would help "
    "an AI assistant do this task well."
)

EDA_LIST = "Continue this list with one new prompt.\n\n{prompts}\n\n" + BARE

# sorted worst to best but labelled best to worst, on purpose (sec 3.2.2)
EDA_RANK = (
    "INSTRUCTION: {mutation_prompt}\n A List of Responses in descending "
    "order of score. {best} is the best response. It resembles {n} more "
    "than it does (1)\n\n{prompts}\n\n" + BARE
)

LINEAGE = (
    "GENOTYPES FOUND IN ASCENDING ORDER OF QUALITY\n\n{prompts}\n\n" + BARE
)

# zero-order hyper-mutation, a new mutation prompt from a thinking style
NEW_MUTATION_PROMPT = (
    "{description} {style}\n\nWrite one instruction for how to improve a "
    "prompt for this task. Return only the instruction."
)

# first-order hyper-mutation
IMPROVE_MUTATION_PROMPT = (
    "Please summarize and improve the following instruction: "
    "{mutation_prompt}\n\nReturn only the improved instruction."
)

# working out to task-prompt
LAMARCKIAN = (
    "I gave a friend an instruction and some advice. Here are the correct "
    "examples of his workings out {example} The instruction was:\n\nDon't "
    "repeat the workings out. Write the general instruction the friend was "
    "given, in one or two sentences. " + BARE
)

# one solver attempt, for lamarckian and reflective
EXAMPLE = "Input:\n{inputs}\n\nResponse:\n{output}"
REFLECT_EXAMPLE = "{example}\nFeedback: {feedback}"

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
