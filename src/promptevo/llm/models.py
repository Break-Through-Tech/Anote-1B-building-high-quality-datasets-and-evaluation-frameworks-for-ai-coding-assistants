"""Chat models. Ollama for both the solver and the mutator, or a vLLM server
for the solver (much faster, see the README)."""

import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import ollama

from promptevo.llm.cache import SqliteCache


class ChatModel(ABC):
    """a chat model, the solver or the mutator. subclasses set name and
    parallel, and implement _send."""

    parallel = 1  # requests in flight at once in chat_many

    def __init__(self):
        self.calls = 0  # chat calls so far
        self._lock = threading.Lock()

    def chat(
        self, user: str, system: str | None = None, seed: int | None = None
    ) -> str:
        """one user message + optional system prompt. returns the reply.
        seed overrides the model's seed for this call."""
        messages = [{"role": "user", "content": user}]
        if system is not None:
            messages.insert(0, {"role": "system", "content": system})
        with self._lock:
            self.calls += 1
        return self._send(messages, seed)

    def chat_many(
        self, users: list[str], system: str | None = None
    ) -> list[str]:
        """chat() for every user message, same system prompt. replies come
        back in order. vllm batches the ones sent together."""
        if self.parallel <= 1:
            return [self.chat(user, system) for user in users]
        with ThreadPoolExecutor(self.parallel) as pool:
            return list(pool.map(lambda user: self.chat(user, system), users))

    # messages in the chat api's format, the reply's text back
    @abstractmethod
    def _send(self, messages: list[dict], seed: int | None) -> str: ...


class OllamaModel(ChatModel):
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
        super().__init__()
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

    def _send(self, messages, seed):
        options = dict(self.options)
        if seed is not None:
            options["seed"] = seed
        reply = self._client.chat(
            model=self.name, messages=messages, options=options
        )
        return reply.message.content


class VLLMModel(ChatModel):
    """a model on a `vllm serve` server, through its openai-style api."""

    # vllm batches whatever is in flight, so send a lot at once
    parallel = 64

    def __init__(
        self,
        name: str,
        url: str = "http://localhost:8000",
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int | None = None,
        repetition_penalty: float = 1.0,
        timeout: float = 1200,
        cache: str | Path | None = None,
    ):
        super().__init__()
        self.name = name
        url = url.rstrip("/")
        self.url = url + "/v1/chat/completions"
        # everything sent every time. anything left out gets the model's
        # generation_config.json default (qwen's has a 1.1 repetition
        # penalty), so results would depend on how the server was started
        self.options = {
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
        }
        if seed is not None:
            self.options["seed"] = seed
        if max_tokens is not None:
            self.options["max_tokens"] = max_tokens
        self._client = httpx.Client(
            timeout=timeout,
            # httpx allows 100 connections by default, more would queue
            limits=httpx.Limits(max_connections=self.parallel),
        )
        # the weights behind the name go in every cache key too
        self._served = served_model(self._client, url, name)
        self._cache = None
        if cache:
            self._cache = SqliteCache(cache, table="replies", column="reply")

    def _send(self, messages, seed):
        body = {"model": self.name, "messages": messages, **self.options}
        if seed is not None:
            body["seed"] = seed
        if self._cache:
            key = _key({"served": self._served, **body})
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        reply = self._post(body)
        if reply.is_error:
            # vllm puts the reason in the body, e.g. prompt too long
            raise RuntimeError(f"vllm {reply.status_code}: {reply.text}")
        text = reply.json()["choices"][0]["message"]["content"]
        if self._cache:
            self._cache.put({key: text})
        return text

    # a dropped connection gets a few more tries before the run loses a
    # whole prompt's score. greedy on a batch invariant server, so a retry
    # gets the same reply
    def _post(self, body, tries=4):
        for attempt in range(tries):
            try:
                return self._client.post(self.url, json=body)
            except httpx.TransportError:
                if attempt == tries - 1:
                    raise
                time.sleep(2 ** attempt)


# checked up front, so a wrong name fails now and not after the seeds have
# been scored
def served_model(client: httpx.Client, url: str, name: str) -> dict:
    """{"root", "len"} of the model the vllm server at url serves under
    name. root is the weights it loaded."""
    served = client.get(url + "/v1/models").json()["data"]
    model = next((m for m in served if m["id"] == name), None)
    if model is None:
        raise ValueError(
            f"{url} serves {[m['id'] for m in served]}, "
            f"not {name}. --solver has to match --served-model-name"
        )
    return {"root": model["root"], "len": model["max_model_len"]}


# reply cache key. the cache is only right for a deterministic (batch
# invariant) server, and the server's flags aren't in the key, so start a
# new cache file if they change
def _key(request):
    text = json.dumps(request, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def solver(
    name: str,
    vllm_url: str | None = None,
    cache: str | Path | None = None,
    max_tokens: int = 1024,
) -> ChatModel:
    """model that fixes the code. greedy, room for whole files. on the vllm
    server at vllm_url if given, else ollama. cache is a reply cache file
    and max_tokens the longest reply, both for vllm."""
    if vllm_url:
        # a bare fix is far shorter than 1024. a longer reply is the model
        # stuck repeating itself, and it holds up the whole batch while it
        # decodes alone. formats that reason first need more
        return VLLMModel(
            name,
            vllm_url,
            temperature=0,
            seed=0,
            max_tokens=max_tokens,
            cache=cache,
        )
    if cache:
        raise ValueError("the reply cache needs a vllm solver")
    return OllamaModel(
        name,
        temperature=0,     # same prompt always gets the same score
        seed=0,
        num_ctx=8192,      # ollama's 4k default can't fit whole files
        num_predict=4096,  # stops runaway output
    )


def mutator(name: str) -> OllamaModel:
    """model that edits prompts."""
    # temp > 0 on purpose, want variety here. reflective mutation shows it
    # whole attempts, more than ollama's 4k default fits
    return OllamaModel(name, temperature=0.8, num_ctx=8192)
