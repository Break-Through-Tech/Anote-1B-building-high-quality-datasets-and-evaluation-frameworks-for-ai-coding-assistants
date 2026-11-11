"""What every dataset has to provide."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from promptevo.models import OllamaModel


# Using dataclasses here instead of NamedTuple for inheritance reasons
# (a NamedTuple subclass can't add fields)
#
# @dataclass writes __init__, __repr__ and __eq__ from the `name: type`
# lines, so ItemResult(tests={...}) just works. frozen=True makes it
# read-only like a tuple

@dataclass(frozen=True)
class ItemResult:
    """one item's tests after the solver's attempt. every test counts, like
    write-from-scratch problems (e.g. APPS)."""

    # default_factory gives each result its own list. a plain `= []` isn't
    # allowed, every result would share one list
    tests:    dict[str, bool] # {test: passed?}. empty = didn't run
    problems: list[str] = field(default_factory=list) # e.g. "syntax error"

    # @property is a method you read like a field, result.target (no parens).
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

    # these come after tests and problems in __init__. they need defaults
    # bc problems has one, so pass them by name:
    # RepairResult(tests=..., to_fix=..., passing=...)
    to_fix:  frozenset[str] = frozenset()  # tests the buggy code fails
    passing: frozenset[str] = frozenset()  # tests the buggy code passes

    # same two properties as ItemResult, narrowed to the bug's tests
    @property
    def target(self):
        return self.to_fix

    @property
    def keep(self):
        return self.passing


class Dataset(ABC):
    """a set of items with tests. subclasses take (items, data_dir=None) so
    the cli can build them."""

    @abstractmethod
    def run(self, prompt: str, solver: OllamaModel) -> dict[str, ItemResult]:
        """solver attempts every item with prompt as its system prompt, then
        the item's tests run. returns {item name: ItemResult}."""
