"""Deep immutability for report objects.

Every report model derives from `Frozen`: fields cannot be reassigned, lists are
stored as tuples and dicts as `FrozenDict`, recursively. The only way to change
a report is to derive a new one (`evolve`), which is what answer / verify /
interpret do — so every change goes through a new revision with a history entry.
"""

from __future__ import annotations

import copy
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, field_validator

_MSG = ("prefit report objects are immutable; use report.answer(), report.verify() or "
        "report.interpret() to derive a new revision")


class FrozenDict(dict):
    """A dict that refuses mutation. Serializes, pickles and deep-copies like a dict."""

    def _readonly(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(_MSG)

    __setitem__ = __delitem__ = _readonly
    clear = pop = popitem = setdefault = update = _readonly
    __ior__ = _readonly

    def __reduce__(self) -> tuple[Any, ...]:
        return (FrozenDict, (dict(self),))

    def __copy__(self) -> FrozenDict:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenDict:
        return FrozenDict({k: copy.deepcopy(v, memo) for k, v in self.items()})

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(tuple(sorted((str(k), repr(v)) for k, v in self.items())))


def freeze(x: Any) -> Any:
    if isinstance(x, FrozenDict):
        return x
    if isinstance(x, dict):
        return FrozenDict({k: freeze(v) for k, v in x.items()})
    if isinstance(x, (list, tuple)):
        return tuple(freeze(v) for v in x)
    return x


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)

    @field_validator("*", mode="after")
    @classmethod
    def _deep_freeze(cls, v: Any) -> Any:
        return freeze(v)


M = TypeVar("M", bound=BaseModel)


def evolve(model: M, **changes: Any) -> M:
    """A validated (and therefore frozen) copy of `model` with some fields replaced.
    Private attributes are not carried over."""
    data = {name: getattr(model, name) for name in type(model).model_fields}
    data.update(changes)
    return type(model).model_validate(data)
