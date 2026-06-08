from __future__ import annotations

from collections.abc import Callable, ItemsView, Iterator, ValuesView
from pathlib import Path
from typing import Literal, Optional, TypeVar, Union, cast

from pydantic import TypeAdapter

from .dbm_sqlite import _Database  # type: ignore
from .dbm_sqlite import _ItemsView as _RawItemsView  # type: ignore
from .dbm_sqlite import _ValuesView as _RawValuesView  # type: ignore
from .query import Expr, Field, compile_expr

T = TypeVar("T")
PathLike = Union[str, Path]
FlagOptions = Literal["r", "w", "c", "n"]


class _ValuesView(_RawValuesView[T]):
    """ValuesView that validates each raw row into the storage_format model."""

    def __iter__(self) -> Iterator[T]:
        mapping: PydanticDBM[T] = self._mapping  # type: ignore
        for raw in super().__iter__():
            yield mapping.type_adapter.validate_json(raw)  # type: ignore

    def __contains__(self, value: object) -> bool:
        return any(item == value for item in self)


class _ItemsView(_RawItemsView[T]):
    """ItemsView that validates each raw row into the storage_format model."""

    def __iter__(self) -> Iterator[tuple[str, T]]:
        mapping: PydanticDBM[T] = self._mapping  # type: ignore
        for key, raw in super().__iter__():
            yield key, mapping.type_adapter.validate_json(raw)  # type: ignore


class _PredicateValuesView(ValuesView[T]):
    """Live view filtered by running a Python predicate over deserialised values.

    Unlike `_ValuesView` (backed by a single SQL query), this re-iterates
    `mapping.values()` and evaluates `predicate` in Python on each pass —
    so it's re-evaluated live, but every record is fetched and deserialised
    rather than filtered in SQLite.
    """

    def __init__(
        self, mapping: PydanticDBM[T], predicate: Callable[[T], Union[bool, int]]
    ) -> None:
        super().__init__(mapping)
        self._predicate = predicate

    def __iter__(self) -> Iterator[T]:
        mapping: PydanticDBM[T] = self._mapping  # type: ignore
        for value in mapping.values():
            if self._predicate(value):
                yield value

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __contains__(self, value: object) -> bool:
        return any(item == value for item in self)


class PydanticDBM(_Database[T]):
    default_storage_format = None

    def __class_getitem__(cls, storage_format: type[T]):
        class _child(cls):
            default_storage_format = storage_format

        _child.__name__ = cls.__name__

        return _child

    def __init__(
        self,
        path: PathLike,
        /,
        *,
        flag: FlagOptions = "c",
        mode: int = 0o600,
        storage_format: Optional[type[T]] = None,
    ):
        super().__init__(path, flag=flag, mode=mode)
        self.storage_format = storage_format or self.default_storage_format
        if not self.storage_format:
            raise ValueError(
                "storage_format must be provided either as argument or class attribute"
            )
        self.type_adapter = TypeAdapter(self.storage_format)

    def __getitem__(self, key: str) -> T:
        return self.type_adapter.validate_json(super().__getitem__(key))  # type: ignore

    def __setitem__(self, key: str, value: T) -> None:
        super().__setitem__(key, self.type_adapter.dump_json(value))  # type: ignore

    def values(self) -> ValuesView[T]:
        return _ValuesView(self)

    def items(self) -> ItemsView[str, T]:
        return _ItemsView(self)

    def query(
        self,
        predicate: Callable[[T], Union[bool, int]],
        *,
        mode: Literal["sql", "filter"] = "sql",
    ) -> ValuesView[T]:
        """Filter stored objects using a predicate on the model.

        Examples::

            db.query(lambda u: u.age >= 30)
            db.query(lambda u: (u.age >= 30) & (u.score > 90))
            db.query(lambda u: u.name.startswith("A") | ~(u.active == True))
            db.query(lambda u: is_in(u.id, [1, 2, 3]))

        ``mode="sql"`` (default): the lambda is type-checked and autocompleted
        as if `u` were a real `storage_format` instance, but is actually run
        once against a `Field` stand-in that builds an expression tree instead
        of evaluating — translated into a single parameterised SQL query, so
        filtering happens in SQLite rather than over every deserialised record.
        Limited to what `Field` can express: comparisons, `&`/`|`/`~`,
        `.contains`/`.startswith`/`.endswith`, and the `is_in()` helper
        (see below).

        Use `&` / `|` / `~` for AND / OR / NOT in this mode, and the
        `pydantic_store.is_in(field, values)` helper for membership tests
        (matches Django's `__in=[...]` convention) — e.g. `is_in(u.id, [1, 2, 3])`
        for `u.id in [1, 2, 3]`. Python can't overload `and`/`or`/`not`/`in`/
        `is`/chained comparisons (they all force a real `bool` — or, for `is`,
        identity — before your code ever runs), which would silently discard
        the expression tree rather than build it; using them raises `TypeError`
        instead. `&`/`|` bind tighter than comparisons, so combined conditions
        need parentheses. For the same reason, a bare field (`m.active`) can't
        be used directly as a condition — write `m.active == True`.

        ``mode="filter"``: the lambda runs as an ordinary Python predicate
        against each deserialised `storage_format` instance — `and`/`or`/`not`
        and arbitrary Python logic (string/collection methods, computed
        properties, multi-field comparisons SQL can't express, ...) all work
        as normal. The cost is that every record is fetched and deserialised,
        with the predicate evaluated in Python rather than pushed into SQLite.

        Returns a live view, re-evaluated on each iteration — re-running it
        reflects the database's current state.
        """
        if mode == "filter":
            return _PredicateValuesView(self, predicate)

        expr = cast(Expr, predicate(cast(T, Field())))
        condition = compile_expr(expr)
        sql = f"SELECT value FROM Dict WHERE {condition.sql}"
        return _ValuesView(self, sql, condition.params)


def open(
    filename: PathLike,
    /,
    flag: FlagOptions = "c",
    mode: int = 0o600,
    storage_format: type[T] = str,
) -> PydanticDBM[T]:
    return PydanticDBM(filename, flag=flag, mode=mode, storage_format=storage_format)
