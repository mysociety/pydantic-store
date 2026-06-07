from __future__ import annotations

from collections.abc import ItemsView, Iterator, ValuesView
from pathlib import Path
from typing import Literal, Optional, TypeVar, Union

from pydantic import TypeAdapter

from .dbm_sqlite import ITER_VALUES, _Database
from .dbm_sqlite import _ItemsView as _RawItemsView
from .dbm_sqlite import _ValuesView as _RawValuesView

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


def open(
    filename: PathLike,
    /,
    flag: FlagOptions = "c",
    mode: int = 0o600,
    storage_format: type[T] = str,
) -> PydanticDBM[T]:
    return PydanticDBM(filename, flag=flag, mode=mode, storage_format=storage_format)
