"""
Helper models and backports
"""

from __future__ import annotations

__version__ = "0.1.0"

import json
from collections.abc import MutableMapping, MutableSequence
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Protocol,
    TypeVar,
    Union,
    overload,
)

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, PrivateAttr
from pydantic import RootModel as PydanticRootModel
from typing_extensions import Self

# Optional imports for YAML and TOML support
try:
    from ruamel.yaml import YAML  # type: ignore

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096  # Avoid line wrapping
except ImportError:
    yaml = None  # type: ignore

try:
    import tomllib  # type: ignore
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore

try:
    import tomli_w  # type: ignore
except ImportError:
    tomli_w = None  # type: ignore

ExtraValues = Literal["allow", "ignore", "forbid"]
FileFormats = Literal["json", "yaml", "toml", "auto"]


T = TypeVar("T")
K = TypeVar("K")
PathLike = Union[str, Path]
IncEx = Union[
    set[int],
    set[str],
    Mapping[int, Union["IncEx", bool]],
    Mapping[str, Union["IncEx", bool]],
]


def file_format_from_file(file_path: Path) -> FileFormats:
    ext = file_path.suffix.lower()
    if ext == ".json":
        return "json"
    elif ext in {".yaml", ".yml"}:
        return "yaml"
    elif ext == ".toml":
        return "toml"
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


class BaseModelLike(Protocol):
    def model_dump(
        self,
        *,
        include: IncEx | None = None,
        exclude: IncEx | None = None,
        context: Any | None = None,
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool = True,
        serialize_as_any: bool = False,
    ) -> dict[str, Any]: ...

    def model_dump_json(
        self,
        *,
        indent: int | None = None,
        ensure_ascii: bool = False,
        include: IncEx | None = None,
        exclude: IncEx | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        exclude_computed_fields: bool = False,
        round_trip: bool = False,
        warnings: bool | Literal["none", "warn", "error"] = True,
        fallback: Callable[[Any], Any] | None = None,
        serialize_as_any: bool = False,
    ) -> str: ...

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self: ...

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self: ...


TypeBaseModelLike = TypeVar("TypeBaseModelLike", bound=BaseModelLike)


def yaml_to_json(yaml_str: str) -> str:
    """
    Convert a YAML string to a JSON string.

    Args:
        yaml_str: YAML string to convert

    Returns:
        JSON string representation

    Raises:
        ImportError: If ruamel.yaml is not available
    """
    if yaml is None:
        raise ImportError(
            "ruamel.yaml is required for YAML support. Install it with: pip install ruamel.yaml"
        )
    parsed_data = yaml.load(yaml_str)  # type: ignore
    return json.dumps(parsed_data)


def toml_to_json(toml_str: str) -> str:
    """
    Convert a TOML string to a JSON string.

    Args:
        toml_str: TOML string to convert

    Returns:
        JSON string representation

    Raises:
        ImportError: If tomllib/tomli is not available
    """
    if tomllib is None:
        raise ImportError(
            "tomllib (Python 3.11+) or tomli is required for TOML reading support. "
            "For Python < 3.11, install with: pip install tomli"
        )
    parsed_data = tomllib.loads(toml_str)  # type: ignore
    return json.dumps(parsed_data)


class IOMixin:
    def to_file(
        self: BaseModelLike, file_path: PathLike, file_format: FileFormats = "auto"
    ) -> None:
        """
        Save the model to a file in the specified format.

        Args:
            file_path: Path to the output file.
            file_format: Format to save the file in ("json", "yaml", or "toml").

        Raises:
            ImportError: If the required library for the format is not installed.
            ValueError: If an unsupported file format is specified.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_format == "auto":
            file_format = file_format_from_file(file_path)

        if file_format == "json":
            txt = self.model_dump_json(indent=2)
        elif file_format == "yaml":
            if yaml is None:
                raise ImportError(
                    "ruamel.yaml is required for YAML support. Install it with: pip install ruamel.yaml"
                )
            # Convert to dict first, then to YAML
            data = self.model_dump()
            from io import StringIO

            stream = StringIO()
            yaml.dump(data, stream)  # type: ignore
            txt = stream.getvalue()
        elif file_format == "toml":
            if tomli_w is None:
                raise ImportError(
                    "tomli-w is required for TOML writing support. Install it with: pip install tomli-w"
                )
            # Convert to dict first, then to TOML
            data = self.model_dump()
            # TOML requires a table (dict) at the root level, not arrays
            if isinstance(data, list):
                raise ValueError(
                    "TOML format does not support arrays at the root level. "
                    "Use JSON or YAML format for list/array data."
                )
            txt = tomli_w.dumps(data)  # type: ignore
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        file_path.write_text(txt)  # type: ignore

    @classmethod
    def from_file(
        cls: type[TypeBaseModelLike],
        file_path: PathLike,
        file_format: FileFormats = "auto",
    ) -> TypeBaseModelLike:
        """
        Load the model from a file in the specified format.

        Args:
            file_path: Path to the input file.
            file_format: Format of the input file ("json", "yaml", or "toml").

        Returns:
            An instance of the model.

        Raises:
            ImportError: If the required library for the format is not installed.
            ValueError: If an unsupported file format is specified.
        """
        file_path = Path(file_path)

        if file_format == "auto":
            file_format = file_format_from_file(file_path)
        if file_format == "json":
            json_data = file_path.read_text()
        elif file_format == "yaml":
            yaml_data = file_path.read_text()
            json_data = yaml_to_json(yaml_data)
        elif file_format == "toml":
            toml_data = file_path.read_text()
            json_data = toml_to_json(toml_data)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        return cls.model_validate_json(json_data)


class BaseModel(PydanticBaseModel, IOMixin): ...


class RootModel(PydanticRootModel[T], IOMixin): ...


class ListModel(RootModel[list[T]], MutableSequence[T]):
    root: list[T] = Field(default_factory=list)  # type: ignore

    def append(self, value: T) -> None:
        self.root.append(value)

    def extend(self, values: Iterable[T]) -> None:
        self.root.extend(values)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> list[T]: ...

    def __getitem__(self, index: int | slice) -> T | list[T]:
        return self.root[index]

    @overload
    def __setitem__(self, index: int, value: T) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...

    def __setitem__(self, index: int | slice, value: T | Iterable[T]) -> None:
        self.root[index] = value  # type: ignore

    def __len__(self) -> int:
        return len(self.root)

    def __iter__(self):  # type: ignore
        return iter(self.root)

    def __contains__(self, item: object) -> bool:
        return item in self.root

    def insert(self, index: int, value: T) -> None:
        self.root.insert(index, value)

    def __delitem__(self, index: int | slice) -> None:
        del self.root[index]


class DictModel(RootModel[dict[K, T]], MutableMapping[K, T]):
    root: dict[K, T] = Field(default_factory=dict)  # type: ignore

    def __contains__(self, key: object) -> bool:
        return key in self.root

    def items(self):
        return self.root.items()

    def keys(self):
        return self.root.keys()

    def values(self):
        return self.root.values()

    def __getitem__(self, key: K) -> T:
        return self.root[key]

    def __setitem__(self, key: K, value: T) -> None:
        self.root[key] = value

    def __delitem__(self, key: K) -> None:
        del self.root[key]

    def __len__(self) -> int:
        return len(self.root)

    def __iter__(self):  # type: ignore
        return iter(self.root)


class JsonStore(DictModel[str, T]):
    _file_path: Optional[Path] = PrivateAttr(default=None)

    def __setitem__(self, key: str, value: T) -> None:
        self.root[key] = value
        self.save_store()

    def __init__(
        self,
        *args: Any,
        file_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._file_path = file_path

    def save_store(self) -> None:
        if self._file_path is None:
            raise ValueError("File path is not set.")
        self.to_file(self._file_path)

    @classmethod
    def connect(cls, file_path: Path) -> JsonStore[T]:
        if not file_path.exists():
            obj = cls(file_path=file_path)
            obj.save_store()
        obj = cls.from_file(file_path)
        obj._file_path = file_path
        return obj
