"""
Test suite for pydantic_store.dbm module.
"""

from collections.abc import ItemsView, ValuesView
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel, Field

from pydantic_store import dbm, is_in
from pydantic_store.dbm import PydanticDBM


# Test models for testing purposes
class SimpleModel(BaseModel):
    name: str
    value: int
    active: bool = True


class NestedModel(BaseModel):
    id: int
    data: SimpleModel
    tags: List[str] = Field(default_factory=list)


class DictModel(BaseModel):
    mapping: Dict[str, Any]
    metadata: Dict[str, str] = Field(default_factory=dict)


class OptionalModel(BaseModel):
    name: str
    maybe: Optional[int] = None


class TestPydanticDBM:
    """Test cases for PydanticDBM class."""

    def test_init_with_storage_format_argument(self, tmp_path: Path):
        """Test initialization with storage_format as argument."""
        db_path = tmp_path / "test.db"

        with PydanticDBM(db_path, storage_format=SimpleModel) as db:
            # Test that we can store and retrieve a model (indicates correct setup)
            model = SimpleModel(name="test", value=42)
            db["test_key"] = model
            retrieved = db["test_key"]
            assert retrieved == model

    def test_init_with_class_subscription(self, tmp_path: Path):
        """Test initialization using class subscription syntax."""
        db_path = tmp_path / "test.db"

        with PydanticDBM[SimpleModel](db_path) as db:
            # Test that we can store and retrieve a model (indicates correct setup)
            model = SimpleModel(name="test", value=42)
            db["test_key"] = model
            retrieved = db["test_key"]
            assert retrieved == model

    def test_init_without_storage_format_raises_error(self, tmp_path: Path):
        """Test that initialization without storage_format raises ValueError."""
        db_path = tmp_path / "test.db"

        with pytest.raises(ValueError, match="storage_format must be provided"):
            PydanticDBM(db_path)

    def test_storage_format_argument_overrides_class_subscription(self, tmp_path: Path):
        """Test that explicit storage_format argument overrides class subscription."""
        db_path = tmp_path / "test.db"

        # Use Any type for generic testing of override behavior
        with PydanticDBM(db_path, storage_format=NestedModel) as db:
            # Test that we can store and retrieve a NestedModel (indicates correct override)
            nested_data = SimpleModel(name="nested", value=123)
            model = NestedModel(id=1, data=nested_data, tags=["tag1"])
            db["test_key"] = model
            retrieved = db["test_key"]
            assert retrieved == model

    def test_setitem_and_getitem_simple_model(self, tmp_path: Path):
        """Test storing and retrieving simple Pydantic models."""
        db_path = tmp_path / "test.db"

        with PydanticDBM[SimpleModel](db_path) as db:
            model = SimpleModel(name="test", value=42, active=False)
            db["key1"] = model

            retrieved = db["key1"]
            assert retrieved == model
            assert isinstance(retrieved, SimpleModel)
            assert retrieved.name == "test"
            assert retrieved.value == 42
            assert retrieved.active is False

    def test_setitem_and_getitem_nested_model(self, tmp_path: Path):
        """Test storing and retrieving nested Pydantic models."""
        db_path = tmp_path / "test.db"

        with PydanticDBM[NestedModel](db_path) as db:
            nested_data = SimpleModel(name="nested", value=123)
            model = NestedModel(id=1, data=nested_data, tags=["tag1", "tag2"])
            db["nested_key"] = model

            retrieved = db["nested_key"]
            assert retrieved == model
            assert isinstance(retrieved, NestedModel)
            assert retrieved.id == 1
            assert retrieved.data.name == "nested"
            assert retrieved.data.value == 123
            assert retrieved.tags == ["tag1", "tag2"]

    def test_setitem_and_getitem_dict_model(self, tmp_path: Path):
        """Test storing and retrieving models with dictionaries."""
        db_path = tmp_path / "test.db"

        with PydanticDBM[DictModel](db_path) as db:
            model = DictModel(
                mapping={"key1": "value1", "key2": 42, "key3": [1, 2, 3]},
                metadata={"author": "test", "version": "1.0"},
            )
            db["dict_key"] = model

            retrieved = db["dict_key"]
            assert retrieved == model
            assert isinstance(retrieved, DictModel)
            assert retrieved.mapping["key1"] == "value1"
            assert retrieved.mapping["key2"] == 42
            assert retrieved.mapping["key3"] == [1, 2, 3]
            assert retrieved.metadata["author"] == "test"

    def test_getitem_nonexistent_key_raises_keyerror(self, tmp_path: Path):
        """Test that accessing non-existent key raises KeyError."""
        db_path = tmp_path / "test.db"

        with PydanticDBM[SimpleModel](db_path) as db:
            with pytest.raises(KeyError):
                _ = db["nonexistent"]

    def test_delitem(self, tmp_path: Path):
        """Test deleting items from the database."""
        db_path = tmp_path / "test.db"

        with PydanticDBM[SimpleModel](db_path) as db:
            model = SimpleModel(name="test", value=42)
            db["key1"] = model

            # Verify it exists
            assert db["key1"] == model

            # Delete it
            del db["key1"]

            # Verify it's gone
            with pytest.raises(KeyError):
                _ = db["key1"]

    def test_delitem_nonexistent_key_raises_keyerror(self, tmp_path: Path):
        """Test that deleting non-existent key raises KeyError."""
        db_path = tmp_path / "test.db"

        with PydanticDBM[SimpleModel](db_path) as db:
            with pytest.raises(KeyError):
                del db["nonexistent"]

    def test_len(self, tmp_path: Path):
        """Test len() function on database."""
        db_path = tmp_path / "test.db"

        with PydanticDBM[SimpleModel](db_path) as db:
            assert len(db) == 0

            db["key1"] = SimpleModel(name="test1", value=1)
            assert len(db) == 1

            db["key2"] = SimpleModel(name="test2", value=2)
            assert len(db) == 2

            del db["key1"]
            assert len(db) == 1

    def test_iteration(self, tmp_path: Path):
        """Test iteration over database keys."""
        db_path = tmp_path / "test.db"

        with PydanticDBM[SimpleModel](db_path) as db:
            keys = ["key1", "key2", "key3"]
            for i, key in enumerate(keys):
                db[key] = SimpleModel(name=f"test{i}", value=i)

            retrieved_keys = list(db)
            assert set(retrieved_keys) == set(keys)

    def test_keys_method(self, tmp_path: Path):
        """Test keys() method."""
        db_path = tmp_path / "test.db"

        with PydanticDBM[SimpleModel](db_path) as db:
            keys = ["key1", "key2", "key3"]
            for i, key in enumerate(keys):
                db[key] = SimpleModel(name=f"test{i}", value=i)

            retrieved_keys = db.keys()
            assert set(retrieved_keys) == set(keys)

    def test_contains(self, tmp_path: Path):
        """Test 'in' operator (contains)."""
        db_path = tmp_path / "test.db"

        with PydanticDBM[SimpleModel](db_path) as db:
            model = SimpleModel(name="test", value=42)
            db["key1"] = model

            assert "key1" in db
            assert "nonexistent" not in db

    def test_persistence_across_sessions(self, tmp_path: Path):
        """Test that data persists across database sessions."""
        db_path = tmp_path / "test.db"
        model = SimpleModel(name="persistent", value=999)

        # First session: write data
        with PydanticDBM[SimpleModel](db_path) as db:
            db["persist_key"] = model

        # Second session: read data
        with PydanticDBM[SimpleModel](db_path) as db:
            retrieved = db["persist_key"]
            assert retrieved == model
            assert retrieved.name == "persistent"
            assert retrieved.value == 999

    def test_different_flag_modes(self, tmp_path: Path):
        """Test different flag modes (r, w, c, n)."""
        db_path = tmp_path / "test.db"
        model = SimpleModel(name="test", value=42)

        # Create with 'c' mode
        with PydanticDBM[SimpleModel](db_path, flag="c") as db:
            db["key1"] = model

        # Read with 'r' mode
        with PydanticDBM[SimpleModel](db_path, flag="r") as db:
            retrieved = db["key1"]
            assert retrieved == model

            # Should not be able to write in read mode
            with pytest.raises(Exception):  # SQLite error
                db["key2"] = model

        # Write with 'w' mode
        with PydanticDBM[SimpleModel](db_path, flag="w") as db:
            db["key2"] = SimpleModel(name="write_test", value=123)

        # Verify both keys exist
        with PydanticDBM[SimpleModel](db_path) as db:
            assert "key1" in db
            assert "key2" in db

    def test_new_flag_mode_clears_existing(self, tmp_path: Path):
        """Test that 'n' flag mode clears existing data."""
        db_path = tmp_path / "test.db"

        # Create initial data
        with PydanticDBM[SimpleModel](db_path, flag="c") as db:
            db["key1"] = SimpleModel(name="test", value=42)

        # Open with 'n' flag (should clear)
        with PydanticDBM[SimpleModel](db_path, flag="n") as db:
            assert len(db) == 0
            assert "key1" not in db

    def test_context_manager_closes_database(self, tmp_path: Path):
        """Test that context manager properly closes database."""
        db_path = tmp_path / "test.db"

        with PydanticDBM[SimpleModel](db_path) as db:
            db["key1"] = SimpleModel(name="test", value=42)
            # Database should be accessible here
            assert "key1" in db

        # After context manager, accessing should fail
        with pytest.raises(Exception):  # Database closed error
            _ = db["key1"]

    def test_manual_close(self, tmp_path: Path):
        """Test manual close() method."""
        db_path = tmp_path / "test.db"

        db = PydanticDBM[SimpleModel](db_path)
        db["key1"] = SimpleModel(name="test", value=42)

        # Verify it works before closing
        assert "key1" in db

        db.close()

        # After closing, accessing should fail
        with pytest.raises(Exception):  # Database closed error
            _ = db["key1"]

    def test_validation_error_on_invalid_data(self, tmp_path: Path):
        """Test that invalid data raises validation error."""
        db_path = tmp_path / "test.db"

        # Create a database with valid data first, then manually corrupt it using SQLite
        with PydanticDBM[SimpleModel](db_path) as db:
            db["valid"] = SimpleModel(name="test", value=42)

        # Manually corrupt the data using SQLite directly
        import sqlite3

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE Dict SET value = ? WHERE key = ?",
                (b'{"name": "test", "value": "not_an_int"}', b"valid"),
            )

        # Now try to read the corrupted data
        with PydanticDBM[SimpleModel](db_path) as db:
            with pytest.raises(Exception):  # Pydantic validation error
                _ = db["valid"]

    def test_custom_mode_parameter(self, tmp_path: Path):
        """Test custom file mode parameter."""
        db_path = tmp_path / "test.db"

        with PydanticDBM[SimpleModel](db_path, mode=0o644) as db:
            db["key1"] = SimpleModel(name="test", value=42)

        # Check that file was created (mode checking is platform-dependent)
        assert db_path.exists()


class TestDBMOpenFunction:
    """Test cases for the dbm.open function."""

    def test_open_function_basic(self, tmp_path: Path):
        """Test basic usage of dbm.open function."""
        db_path = tmp_path / "test.db"

        with dbm.open(db_path, storage_format=SimpleModel) as db:
            model = SimpleModel(name="test", value=42)
            db["key1"] = model

            retrieved = db["key1"]
            assert retrieved == model
            assert isinstance(retrieved, SimpleModel)

    def test_open_function_with_string_type(self, tmp_path: Path):
        """Test dbm.open with string storage format."""
        db_path = tmp_path / "test.db"

        with dbm.open(db_path, storage_format=str) as db:
            db["key1"] = "test_string"

            retrieved = db["key1"]
            assert retrieved == "test_string"
            assert isinstance(retrieved, str)

    def test_open_function_with_different_flags(self, tmp_path: Path):
        """Test dbm.open with different flag modes."""
        db_path = tmp_path / "test.db"

        # Create with default flag
        with dbm.open(db_path, storage_format=SimpleModel) as db:
            db["key1"] = SimpleModel(name="test", value=42)

        # Open read-only
        with dbm.open(db_path, flag="r", storage_format=SimpleModel) as db:
            retrieved = db["key1"]
            assert retrieved.name == "test"

    def test_open_function_parameters(self, tmp_path: Path):
        """Test dbm.open with all parameters."""
        db_path = tmp_path / "test.db"

        with dbm.open(db_path, flag="c", mode=0o600, storage_format=SimpleModel) as db:
            db["test"] = SimpleModel(name="param_test", value=123)
            assert db["test"].name == "param_test"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_key_string(self, tmp_path: Path):
        """Test using empty string as key."""
        db_path = tmp_path / "test.db"

        with PydanticDBM[SimpleModel](db_path) as db:
            model = SimpleModel(name="empty_key", value=42)
            db[""] = model

            retrieved = db[""]
            assert retrieved == model

    def test_unicode_keys(self, tmp_path: Path):
        """Test using unicode strings as keys."""
        db_path = tmp_path / "test.db"

        with PydanticDBM[SimpleModel](db_path) as db:
            model = SimpleModel(name="unicode", value=42)
            unicode_key = "测试_🔑_key"
            db[unicode_key] = model

            retrieved = db[unicode_key]
            assert retrieved == model

    def test_large_model_storage(self, tmp_path: Path):
        """Test storing large models with lots of data."""
        db_path = tmp_path / "test.db"

        with PydanticDBM[DictModel](db_path) as db:
            large_dict = {f"key_{i}": f"value_{i}" * 100 for i in range(1000)}
            model = DictModel(mapping=large_dict)
            db["large"] = model

            retrieved = db["large"]
            assert retrieved == model
            assert len(retrieved.mapping) == 1000

    def test_model_with_none_values(self, tmp_path: Path):
        """Test models with None/optional values."""

        class NullableModel(BaseModel):
            name: str
            value: Optional[int] = None
            active: bool = True

        db_path = tmp_path / "test.db"

        with PydanticDBM[NullableModel](db_path) as db:
            model = NullableModel(name="test", value=None)
            db["optional"] = model

            retrieved = db["optional"]
            assert retrieved == model
            assert retrieved.name == "test"
            assert retrieved.value is None
            assert retrieved.active is True

    def test_concurrent_access_same_process(self, tmp_path: Path):
        """Test concurrent access from same process (should work with SQLite WAL mode)."""
        db_path = tmp_path / "test.db"

        # Open two connections to same database
        db1 = PydanticDBM[SimpleModel](db_path)
        db2 = PydanticDBM[SimpleModel](db_path)

        try:
            # Write with first connection
            db1["key1"] = SimpleModel(name="from_db1", value=1)

            # Read with second connection
            retrieved = db2["key1"]
            assert retrieved.name == "from_db1"

            # Write with second connection
            db2["key2"] = SimpleModel(name="from_db2", value=2)

            # Read with first connection
            retrieved = db1["key2"]
            assert retrieved.name == "from_db2"
        finally:
            db1.close()
            db2.close()

    def test_reopen_after_close(self, tmp_path: Path):
        """Test that database can be reopened after closing."""
        db_path = tmp_path / "test.db"

        # Create and close database
        db = PydanticDBM[SimpleModel](db_path)
        db["key1"] = SimpleModel(name="test", value=42)
        db.close()

        # Reopen same database file
        db2 = PydanticDBM[SimpleModel](db_path)
        try:
            retrieved = db2["key1"]
            assert retrieved.name == "test"
            assert retrieved.value == 42
        finally:
            db2.close()

    def test_access_after_close_raises_error(self, tmp_path: Path):
        """Test that accessing closed database raises error."""
        db_path = tmp_path / "test.db"

        db = PydanticDBM[SimpleModel](db_path)
        db["key1"] = SimpleModel(name="test", value=42)
        db.close()

        # Accessing closed database should raise error
        with pytest.raises(Exception):  # Database error
            _ = db["key1"]


class TestBulkMethods:
    """Test cases for values() and items() on PydanticDBM."""

    # --- values() ---

    def test_values_returns_view(self, tmp_path: Path):
        models = [
            SimpleModel(name="a", value=1),
            SimpleModel(name="b", value=2),
            SimpleModel(name="c", value=3),
        ]
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            for i, m in enumerate(models):
                db[str(i)] = m
            result = db.values()
            assert isinstance(result, ValuesView)
            assert len(result) == 3
            assert set(m.name for m in result) == {"a", "b", "c"}
            assert SimpleModel(name="a", value=1) in result

    def test_values_is_live(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["a"] = SimpleModel(name="a", value=1)
            view = db.values()
            assert len(view) == 1
            db["b"] = SimpleModel(name="b", value=2)
            assert len(view) == 2
            assert {m.name for m in view} == {"a", "b"}

    def test_values_empty(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            assert list(db.values()) == []

    # --- items() ---

    def test_items_returns_view(self, tmp_path: Path):
        m = SimpleModel(name="hello", value=99)
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["k1"] = m
            result = db.items()
            assert isinstance(result, ItemsView)
            assert len(result) == 1
            assert ("k1", m) in result
            key, value = next(iter(result))
            assert key == "k1"
            assert value == m

    def test_items_is_live(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["a"] = SimpleModel(name="a", value=1)
            view = db.items()
            assert len(view) == 1
            db["b"] = SimpleModel(name="b", value=2)
            assert len(view) == 2
            assert dict((k, v.name) for k, v in view) == {"a": "a", "b": "b"}

    def test_items_empty(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            assert list(db.items()) == []


class TestLambdaQuery:
    """Test cases for PydanticDBM.query() — lambda-based filter expressions."""

    # --- comparisons ---

    def test_query_gt(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["a"] = SimpleModel(name="a", value=1)
            db["b"] = SimpleModel(name="b", value=5)
            db["c"] = SimpleModel(name="c", value=10)
            result = db.query(lambda m: m.value > 4)
            assert {m.name for m in result} == {"b", "c"}

    def test_query_eq(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["a"] = SimpleModel(name="alice", value=42)
            db["b"] = SimpleModel(name="bob", value=7)
            result = db.query(lambda m: m.value == 42)
            assert len(result) == 1
            assert next(iter(result)).name == "alice"

    def test_query_ne(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["a"] = SimpleModel(name="alice", value=42)
            db["b"] = SimpleModel(name="bob", value=7)
            result = db.query(lambda m: m.value != 42)
            assert len(result) == 1
            assert next(iter(result)).name == "bob"

    def test_query_eq_none(self, tmp_path: Path):
        # `is None` can't be overloaded (no __is__ dunder) and always evaluates
        # to a literal False — `== None` is the correct way to match nulls, and
        # relies on `build_condition` compiling it to `IS NULL`, not `= NULL`
        # (SQL's `= NULL` is always NULL/falsy, even for genuinely-null values).
        with PydanticDBM[OptionalModel](tmp_path / "test.db") as db:
            db["a"] = OptionalModel(name="a", maybe=None)
            db["b"] = OptionalModel(name="b", maybe=5)
            result = db.query(lambda m: m.maybe == None)  # noqa: E711
            assert len(result) == 1
            assert next(iter(result)).name == "a"

    def test_query_ne_none(self, tmp_path: Path):
        with PydanticDBM[OptionalModel](tmp_path / "test.db") as db:
            db["a"] = OptionalModel(name="a", maybe=None)
            db["b"] = OptionalModel(name="b", maybe=5)
            result = db.query(lambda m: m.maybe != None)  # noqa: E711
            assert len(result) == 1
            assert next(iter(result)).name == "b"

    def test_query_lt_le_ge(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["a"] = SimpleModel(name="a", value=1)
            db["b"] = SimpleModel(name="b", value=5)
            db["c"] = SimpleModel(name="c", value=10)
            assert {m.name for m in db.query(lambda m: m.value < 5)} == {"a"}
            assert {m.name for m in db.query(lambda m: m.value <= 5)} == {"a", "b"}
            assert {m.name for m in db.query(lambda m: m.value >= 5)} == {"b", "c"}

    # --- boolean combinators ---

    def test_query_and(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["a"] = SimpleModel(name="a", value=10, active=True)
            db["b"] = SimpleModel(name="b", value=3, active=True)
            db["c"] = SimpleModel(name="c", value=10, active=False)
            result = db.query(lambda m: (m.value > 5) & (m.active == True))  # noqa: E712
            assert len(result) == 1
            assert next(iter(result)).name == "a"

    def test_query_or(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["a"] = SimpleModel(name="a", value=1)
            db["b"] = SimpleModel(name="b", value=5)
            db["c"] = SimpleModel(name="c", value=10)
            result = db.query(lambda m: (m.value < 2) | (m.value > 8))
            assert {m.name for m in result} == {"a", "c"}

    def test_query_not(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["a"] = SimpleModel(name="a", value=1, active=True)
            db["b"] = SimpleModel(name="b", value=2, active=False)
            result = db.query(lambda m: ~(m.active == True))  # type: ignore  # noqa: E712
            assert len(result) == 1
            assert next(iter(result)).name == "b"

    def test_query_combined_and_or(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["a"] = SimpleModel(name="a", value=10, active=True)
            db["b"] = SimpleModel(name="b", value=10, active=False)
            db["c"] = SimpleModel(name="c", value=1, active=True)
            result = db.query(
                lambda m: ((m.value > 5) & (m.active == True)) | (m.value < 2)  # type: ignore  # noqa: E712
            )  # type: ignore  # noqa: E712
            assert {m.name for m in result} == {"a", "c"}

    # --- lookup-style methods ---

    def test_query_contains_startswith_endswith(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["a"] = SimpleModel(name="alice", value=1)
            db["b"] = SimpleModel(name="alicia", value=2)
            db["c"] = SimpleModel(name="bob", value=3)
            assert {m.name for m in db.query(lambda m: m.name.startswith("ali"))} == {
                "alice",
                "alicia",
            }
            assert {m.name for m in db.query(lambda m: m.name.endswith("ia"))} == {
                "alicia"
            }

    def test_query_in(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["a"] = SimpleModel(name="a", value=1)
            db["b"] = SimpleModel(name="b", value=2)
            db["c"] = SimpleModel(name="c", value=3)
            result = db.query(lambda m: is_in(m.value, [1, 3]))
            assert {m.name for m in result} == {"a", "c"}

    # --- path navigation ---

    def test_query_nested_path(self, tmp_path: Path):
        with PydanticDBM[NestedModel](tmp_path / "test.db") as db:
            db["a"] = NestedModel(id=1, data=SimpleModel(name="x", value=50))
            db["b"] = NestedModel(id=2, data=SimpleModel(name="y", value=150))
            result = db.query(lambda m: m.data.value > 100)
            assert len(result) == 1
            assert next(iter(result)).id == 2

    # --- return value ---

    def test_query_returns_view(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["a"] = SimpleModel(name="alice", value=42)
            db["b"] = SimpleModel(name="bob", value=7)
            result = db.query(lambda m: m.value > 5)
            assert isinstance(result, ValuesView)
            assert len(result) == 2

    def test_query_is_live(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["a"] = SimpleModel(name="alice", value=42)
            view = db.query(lambda m: m.value > 5)
            assert len(view) == 1
            db["b"] = SimpleModel(name="bob", value=7)
            assert len(view) == 2

    # --- edge cases ---

    def test_query_empty_db(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            assert list(db.query(lambda m: m.value > 0)) == []

    def test_query_no_matches(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["a"] = SimpleModel(name="a", value=1)
            assert list(db.query(lambda m: m.value > 9999)) == []

    def test_query_and_or_not_raise(self, tmp_path: Path):
        # 'and'/'or'/'not' convert their operands to bool before the lambda's
        # logic can run, which would silently drop part of the expression
        # tree (e.g. "(m.a > 1) and (m.b > 2)" collapsing to just "m.b > 2").
        # Using them must raise loudly rather than build a silently-wrong query.
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["a"] = SimpleModel(name="a", value=1)
            with pytest.raises(TypeError, match="&.*\\|.*~"):
                db.query(lambda m: (m.value > 1) and (m.active == True))  # noqa: E712
            with pytest.raises(TypeError, match="&.*\\|.*~"):
                db.query(lambda m: not (m.value > 1))
            with pytest.raises(TypeError, match="&.*\\|.*~"):
                db.query(lambda m: m.active and (m.value > 1))


class TestQueryFilterMode:
    """Test cases for PydanticDBM.query(predicate, mode="filter")."""

    def test_filter_mode_runs_predicate_on_real_instances(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["a"] = SimpleModel(name="alice", value=1)
            db["b"] = SimpleModel(name="bob", value=2)
            result = db.query(lambda m: m.name.upper() == "ALICE", mode="filter")
            assert {m.name for m in result} == {"alice"}

    def test_filter_mode_supports_and_or_not(self, tmp_path: Path):
        # Unlike mode="sql", the predicate runs against real model instances
        # here, so ordinary 'and'/'or'/'not' work exactly as normal Python.
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["a"] = SimpleModel(name="a", value=10, active=True)
            db["b"] = SimpleModel(name="b", value=3, active=True)
            db["c"] = SimpleModel(name="c", value=10, active=False)
            result = db.query(lambda m: m.value > 5 and m.active, mode="filter")
            assert {m.name for m in result} == {"a"}

    def test_filter_mode_bare_field_truthiness(self, tmp_path: Path):
        # mode="sql" can't evaluate a bare field's truthiness (no universal
        # SQL equivalent of Python's truthy/falsy) — mode="filter" can,
        # since the predicate runs against the real deserialised value.
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["a"] = SimpleModel(name="a", value=1, active=True)
            db["b"] = SimpleModel(name="b", value=2, active=False)
            result = db.query(lambda m: m.active, mode="filter")
            assert {m.name for m in result} == {"a"}

    def test_filter_mode_returns_view(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["a"] = SimpleModel(name="alice", value=42)
            db["b"] = SimpleModel(name="bob", value=7)
            result = db.query(lambda m: m.value > 5, mode="filter")
            assert isinstance(result, ValuesView)
            assert len(result) == 2
            assert {m.name for m in result} == {"alice", "bob"}

    def test_filter_mode_is_live(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["a"] = SimpleModel(name="alice", value=42)
            view = db.query(lambda m: m.value > 5, mode="filter")
            assert len(view) == 1
            db["b"] = SimpleModel(name="bob", value=7)
            assert len(view) == 2
            assert {m.name for m in view} == {"alice", "bob"}

    def test_filter_mode_empty_db(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            assert list(db.query(lambda m: m.value > 0, mode="filter")) == []

    def test_filter_mode_no_matches(self, tmp_path: Path):
        with PydanticDBM[SimpleModel](tmp_path / "test.db") as db:
            db["a"] = SimpleModel(name="a", value=1)
            assert list(db.query(lambda m: m.value > 9999, mode="filter")) == []
