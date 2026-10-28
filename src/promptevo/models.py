"""Ollama chat models. One class for both the solver and the mutator."""

import ollama


class OllamaModel:
    """an ollama model + its sampling options."""

    def __init__(
        self,
        name: str,
        temperature: float = 0.0,
        seed: int | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        timeout: float = 1200,
    ):
        self.name = name
        options = {
            "temperature": temperature,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }
        # unset ones fall back to ollama's defaults
        self.options = {k: v for k, v in options.items() if v is not None}
        # a stuck request would hang the EA forever
        self._client = ollama.Client(timeout=timeout)

    def chat(
        self, user: str, system: str | None = None, seed: int | None = None
    ) -> str:
        """one user message + optional system prompt. returns the reply.
        seed overrides the model's seed for this call."""
        messages = [{"role": "user", "content": user}]
        if system is not None:
            messages.insert(0, {"role": "system", "content": system})
        options = dict(self.options)
        if seed is not None:
            options["seed"] = seed
        reply = self._client.chat(
            model=self.name, messages=messages, options=options
        )
        return reply.message.content


def solver(name: str) -> OllamaModel:
    """model that fixes the code. greedy, room for whole files."""
    return OllamaModel(
        name,
        temperature=0,     # same prompt always gets the same score
        seed=0,
        num_ctx=8192,      # ollama's 4k default can't fit whole files
        num_predict=4096,  # stops runaway output
    )


def mutator(name: str) -> OllamaModel:
    """model that edits prompts."""
    # temp > 0 on purpose, want variety here
    return OllamaModel(name, temperature=0.8)