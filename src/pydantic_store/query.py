"""
Lambda-based query expressions for PydanticDBM.query(mode="sql").

A `db.query(lambda x: x.key_a.key_b > 3)` style API can't evaluate the lambda
against a real model instance — there's nothing to evaluate it against until
the matching records are known. Instead, `Field` stands in for the model
inside the lambda: attribute access and comparisons on it build an expression
tree (`Comparison` / `BoolOp` / `Not`) rather than producing a real result.
`compile_expr` then walks that tree and translates it into a parameterised
SQL `WHERE` fragment, via `build_condition`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, NamedTuple, Optional, TypeVar, Union, cast

KNOWN_OPERATORS: frozenset[str] = frozenset(
    {"exact", "gt", "gte", "lt", "lte", "contains", "startswith", "endswith", "in"}
)


class Condition(NamedTuple):
    sql: str
    params: list[object]


def build_condition(
    json_path: str, operator: str, target: object | list[object]
) -> Condition:
    """Return a Condition(sql, params) for one predicate.

    json_path is always passed as a SQLite parameter, never interpolated,
    to prevent injection via kwarg key names.
    """
    if operator == "exact":
        if target is None:
            # SQL's `= NULL` is always NULL (never true), even when the value
            # genuinely is null — `IS NULL` is the only correct comparison.
            return Condition("json_extract(Dict.value, ?) IS NULL", [json_path])
        return Condition("json_extract(Dict.value, ?) = ?", [json_path, target])
    elif operator == "gt":
        return Condition("json_extract(Dict.value, ?) > ?", [json_path, target])
    elif operator == "gte":
        return Condition("json_extract(Dict.value, ?) >= ?", [json_path, target])
    elif operator == "lt":
        return Condition("json_extract(Dict.value, ?) < ?", [json_path, target])
    elif operator == "lte":
        return Condition("json_extract(Dict.value, ?) <= ?", [json_path, target])
    elif operator == "startswith":
        return Condition(
            "json_extract(Dict.value, ?) LIKE ?", [json_path, f"{target}%"]
        )
    elif operator == "endswith":
        return Condition(
            "json_extract(Dict.value, ?) LIKE ?", [json_path, f"%{target}"]
        )
    elif operator == "in":
        if not isinstance(target, list):
            raise ValueError(
                f"Operator 'in' requires a list of target values, not {target!r}"
            )
        placeholders = ",".join("?" * len(target))  # type: ignore[arg-type]
        return Condition(
            f"json_extract(Dict.value, ?) IN ({placeholders})", [json_path, *target]
        )  # type: ignore[misc]
    elif operator == "contains":
        # Handles both string fields (LIKE substring) and array fields (json_each membership).
        # json_type disambiguates at query time so no schema knowledge is required here.
        sql = (
            "((json_type(Dict.value, ?) != 'array' AND json_extract(Dict.value, ?) LIKE ?) "
            "OR (json_type(Dict.value, ?) = 'array' AND "
            "EXISTS (SELECT 1 FROM json_each(Dict.value, ?) WHERE value = ?)))"
        )
        return Condition(
            sql, [json_path, json_path, f"%{target}%", json_path, json_path, target]
        )
    else:
        raise ValueError(
            f"Unknown operator {operator!r}. Valid operators: {sorted(KNOWN_OPERATORS)}"
        )


def _no_bool(self: object) -> bool:
    """Refuse implicit boolean conversion of expressions and `Field`s.

    Several Python constructs — `and` / `or` / `not` / `in` / `is` / chained
    comparisons (`a < b < c`) / `if` — convert their operand(s) to a real
    `bool` before any of our code can run, and there's no dunder hook to
    intercept the conversion itself. Left to the default (truthy) object
    behaviour, e.g. `(x.a > 1) and (x.b > 2)` would silently collapse to
    just `(x.b > 2)`, discarding the first condition without any error.
    Raising here turns that into an immediate, loud failure pointing at
    the offending lambda, telling the user what to use instead.
    """
    raise TypeError(
        "query() expressions can't be evaluated as booleans — this happens "
        "inside 'and'/'or'/'not'/'in'/'is'/'if'/chained comparisons, all of "
        "which force a real bool and would silently discard the expression "
        "tree rather than build it. None of them can be overloaded to avoid "
        "this. Use '&'/'|'/'~' for AND/OR/NOT (e.g. (x.a > 1) & (x.b > 2)), "
        "'is_in(x.a, [...])' for membership (not 'x.a in [...]'), and "
        "'==' / '!=' for equality (not 'is' / 'is not')."
    )


class Comparison(NamedTuple):
    path: list[str]
    operator: str
    value: object

    def __and__(self, other: Expr) -> BoolOp:
        return BoolOp("and", [self, other])

    def __or__(self, other: Expr) -> BoolOp:
        return BoolOp("or", [self, other])

    def __invert__(self) -> Not:
        return Not(self)

    __bool__ = _no_bool


class BoolOp(NamedTuple):
    operator: Literal["and", "or"]
    parts: list[Expr]

    def __and__(self, other: Expr) -> BoolOp:
        return BoolOp("and", [self, other])

    def __or__(self, other: Expr) -> BoolOp:
        return BoolOp("or", [self, other])

    def __invert__(self) -> Not:
        return Not(self)

    __bool__ = _no_bool


class Not(NamedTuple):
    expr: Expr

    def __and__(self, other: Expr) -> BoolOp:
        return BoolOp("and", [self, other])

    def __or__(self, other: Expr) -> BoolOp:
        return BoolOp("or", [self, other])

    def __invert__(self) -> Not:
        return Not(self)

    __bool__ = _no_bool


Expr = Union[Comparison, BoolOp, Not]


class Field:
    """Stand-in for a model instance inside a `query()` lambda.

    Attribute access extends a field path; comparisons and lookup methods
    turn that path into a `Comparison`; `&` / `|` / `~` combine expressions
    into `BoolOp` / `Not` nodes. The lambda never runs against real data —
    it runs once against a `Field` to build an expression tree, which
    `compile_expr` then translates into SQL.
    """

    def __init__(self, path: Optional[list[str]] = None) -> None:
        self._path: list[str] = path if path is not None else []

    def __getattr__(self, name: str) -> Field:
        return Field([*self._path, name])

    def __eq__(self, other: object) -> Comparison:  # type: ignore[override]
        return Comparison(self._path, "exact", other)

    def __ne__(self, other: object) -> Not:  # type: ignore[override]
        return Not(Comparison(self._path, "exact", other))

    def __gt__(self, other: object) -> Comparison:
        return Comparison(self._path, "gt", other)

    def __ge__(self, other: object) -> Comparison:
        return Comparison(self._path, "gte", other)

    def __lt__(self, other: object) -> Comparison:
        return Comparison(self._path, "lt", other)

    def __le__(self, other: object) -> Comparison:
        return Comparison(self._path, "lte", other)

    def contains(self, value: object) -> Comparison:
        return Comparison(self._path, "contains", value)

    def startswith(self, value: object) -> Comparison:
        return Comparison(self._path, "startswith", value)

    def endswith(self, value: object) -> Comparison:
        return Comparison(self._path, "endswith", value)

    __bool__ = _no_bool


ValueT = TypeVar("ValueT")


def is_in(value: ValueT, options: Iterable[ValueT]) -> bool:
    """Membership test for use inside query() lambdas — `is_in(x, [1, 2, 3])`
    reads as "is x in [1, 2, 3]", matching Django's `__in=[...]` convention.

    `in` can't be used directly: like `and`/`or`/`not`/`is`, it forces its
    result through a real `bool()` at the C level (inside the container's
    `__contains__`), which would silently discard the expression tree no
    matter which class defines `__contains__`. `is_in` sidesteps this by
    being an ordinary function — declared (and type-checked) as comparing a
    real value against an iterable of the same type and returning `bool`,
    it actually receives the `Field` stand-in `value` is at runtime and builds
    the equivalent SQL `IN` comparison::

        db.query(lambda m: is_in(m.value, [1, 2, 3]))
    """
    field = cast(Field, value)
    return cast(bool, Comparison(field._path, "in", list(options)))  # type: ignore[return-value]


def compile_expr(expr: Expr) -> Condition:
    """Translate an expression tree into a parameterised SQL WHERE fragment."""
    if isinstance(expr, Comparison):
        json_path = "$." + ".".join(expr.path)
        return build_condition(json_path, expr.operator, expr.value)

    if isinstance(expr, Not):
        inner = compile_expr(expr.expr)
        return Condition(f"NOT ({inner.sql})", inner.params)

    joiner = " AND " if expr.operator == "and" else " OR "
    parts = [compile_expr(part) for part in expr.parts]
    sql = "(" + joiner.join(part.sql for part in parts) + ")"
    params = [param for part in parts for param in part.params]
    return Condition(sql, params)
