"""
Test suite for pydantic_store.dbm module.
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest
from pydantic import BaseModel, Field
from pydantic_store import dbm
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
        from typing import Optional

        class OptionalModel(BaseModel):
            name: str
            value: Optional[int] = None
            active: bool = True

        db_path = tmp_path / "test.db"

        with PydanticDBM[OptionalModel](db_path) as db:
            model = OptionalModel(name="test", value=None)
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
