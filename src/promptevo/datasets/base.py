"""What every dataset has to provide."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from promptevo.llm.models import ChatModel


# dataclasses, not NamedTuple, since a NamedTuple subclass can't add fields.
# frozen because the evaluator caches results and hands the same ones out


@dataclass(frozen=True)
class ItemResult:
    """one item's tests after the solver's attempt. every test counts, like
    write-from-scratch problems (e.g. APPS)."""

    # {test: passed?}, empty if the tests didn't run
    tests:    dict[str, bool]
    problems: list[str] = field(default_factory=list)  # e.g. "syntax error"
    # what the solver was sent and what it replied, for mutations that look
    # at real attempts. empty if the dataset doesn't fill them in
    inputs:   str = ""
    output:   str = ""

    # partial() and binary() only read target and keep, so a subclass changes
    # what counts by overriding these two

    # tests the score counts
    @property
    def target(self) -> frozenset[str]:
        return frozenset(self.tests)

    # tests that must still pass or the item scores 0
    @property
    def keep(self) -> frozenset[str]:
        return frozenset()


@dataclass(frozen=True)
class RepairResult(ItemResult):
    """buggy-code item. only the tests the bug breaks count, the ones the
    buggy code already passes must keep passing."""

    # defaults only because problems has one, so pass these by name
    to_fix:  frozenset[str] = frozenset()  # tests the buggy code fails
    passing: frozenset[str] = frozenset()  # tests the buggy code passes

    # ItemResult's two properties, narrowed to the bug's tests
    @property
    def target(self):
        return self.to_fix

    @property
    def keep(self):
        return self.passing


class Dataset(ABC):
    """a set of items with tests. subclasses take (items, data_dir=None) so
    the cli can build them."""

    # what the task is, in a sentence or two. some mutations show it to the
    # mutator
    description = "Fix the bug in the given code so its tests pass."

    @property
    @abstractmethod
    def names(self) -> list[str]:
        """every item's name, in the order run() goes through them."""

    def __len__(self):
        return len(self.names)

    @abstractmethod
    def run(
        self,
        prompt: str,
        solver: ChatModel,
        items: list[str] | None = None,
    ) -> dict[str, ItemResult]:
        """solver attempts every item (or just items) with prompt as its
        system prompt, then the tests run. returns {item name: ItemResult}."""
