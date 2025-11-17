import json
from pathlib import Path
from typing import Any, Union

import pytest
from pydantic_store.models import (
    BaseModel,
    DictModel,
    JsonStore,
    ListModel,
    toml_to_json,
    yaml_to_json,
)


# Test models for testing purposes
class SimpleModel(BaseModel):
    name: str
    value: int


class StringListModel(ListModel[str]):
    pass


class StringIntDictModel(DictModel[str, int]):
    pass


class ComplexModel(BaseModel):
    items: list[str]
    mapping: dict[str, Any]
    nested: SimpleModel


# BaseModel Tests
class TestBaseModelClass:
    def test_to_file_creates_json(self, tmp_path: Path) -> None:
        """Test that to_file creates a properly formatted JSON file."""
        model = SimpleModel(name="test", value=42)
        file_path = tmp_path / "test.json"

        model.to_file(file_path)

        assert file_path.exists()
        with open(file_path) as f:
            data = json.load(f)
        assert data == {"name": "test", "value": 42}

    def test_to_file_creates_directories(self, tmp_path: Path) -> None:
        """Test that to_file creates parent directories if they don't exist."""
        model = SimpleModel(name="test", value=42)
        file_path = tmp_path / "nested" / "deep" / "test.json"

        model.to_file(file_path)

        assert file_path.exists()
        assert file_path.parent.exists()

    def test_from_file_loads_json(self, tmp_path: Path) -> None:
        """Test that from_file correctly loads a model from JSON."""
        file_path = tmp_path / "test.json"
        data: dict[str, Any] = {"name": "loaded", "value": 100}
        with open(file_path, "w") as f:
            json.dump(data, f)

        model = SimpleModel.from_file(file_path)

        assert model.name == "loaded"
        assert model.value == 100

    def test_roundtrip_file_operations(self, tmp_path: Path) -> None:
        """Test that saving and loading preserves model data."""
        original = SimpleModel(name="roundtrip", value=999)
        file_path = tmp_path / "roundtrip.json"

        original.to_file(file_path)
        loaded = SimpleModel.from_file(file_path)

        assert loaded.name == original.name
        assert loaded.value == original.value
        assert loaded == original

    def test_complex_model_serialization(self, tmp_path: Path) -> None:
        """Test serialization of complex nested models."""
        nested = SimpleModel(name="nested", value=1)
        complex_model = ComplexModel(
            items=["a", "b", "c"], mapping={"key": "value", "number": 42}, nested=nested
        )
        file_path = tmp_path / "complex.json"

        complex_model.to_file(file_path)
        loaded = ComplexModel.from_file(file_path)

        assert loaded.items == complex_model.items
        assert loaded.mapping == complex_model.mapping
        assert loaded.nested.name == complex_model.nested.name
        assert loaded.nested.value == complex_model.nested.value

    def test_yaml_format_support(self, tmp_path: Path) -> None:
        """Test YAML format support with proper error handling."""
        model = SimpleModel(name="yaml_test", value=123)
        file_path = tmp_path / "test.yaml"

        try:
            # Try to save as YAML
            model.to_file(file_path, file_format="yaml")

            # If we get here, PyYAML is available
            assert file_path.exists()

            # Test loading
            loaded = SimpleModel.from_file(file_path, file_format="yaml")
            assert loaded.name == model.name
            assert loaded.value == model.value

        except ImportError as e:
            # If ruamel.yaml is not available, should get a helpful error message
            assert "ruamel.yaml is required for YAML support" in str(e)
            assert "pip install ruamel.yaml" in str(e)

    def test_toml_format_support(self, tmp_path: Path) -> None:
        """Test TOML format support with proper error handling."""
        model = SimpleModel(name="toml_test", value=456)
        file_path = tmp_path / "test.toml"

        try:
            # Try to save as TOML
            model.to_file(file_path, file_format="toml")

            # If we get here, tomli_w is available
            assert file_path.exists()

            # Test loading
            loaded = SimpleModel.from_file(file_path, file_format="toml")
            assert loaded.name == model.name
            assert loaded.value == model.value

        except ImportError as e:
            # If tomli_w or tomllib is not available, should get a helpful error message
            assert (
                "tomli-w is required for TOML writing support" in str(e)
                or "tomllib" in str(e)
                or "tomli is required for TOML reading support" in str(e)
            )

    def test_unsupported_format_raises_error(self, tmp_path: Path) -> None:
        """Test that unsupported formats raise ValueError."""
        model = SimpleModel(name="test", value=789)
        file_path = tmp_path / "test.xml"

        with pytest.raises(ValueError, match="Unsupported file format: xml"):
            model.to_file(file_path, file_format="xml")  # type: ignore

        with pytest.raises(ValueError, match="Unsupported file format: xml"):
            SimpleModel.from_file(file_path, file_format="xml")  # type: ignore

    def test_complex_model_yaml_serialization(self, tmp_path: Path) -> None:
        """Test YAML serialization of complex nested models."""
        nested = SimpleModel(name="nested_yaml", value=1)
        complex_model = ComplexModel(
            items=["yaml", "test", "items"],
            mapping={"yaml_key": "yaml_value", "number": 42},
            nested=nested,
        )
        file_path = tmp_path / "complex.yaml"

        try:
            complex_model.to_file(file_path, file_format="yaml")
            loaded = ComplexModel.from_file(file_path, file_format="yaml")

            assert loaded.items == complex_model.items
            assert loaded.mapping == complex_model.mapping
            assert loaded.nested.name == complex_model.nested.name
            assert loaded.nested.value == complex_model.nested.value
        except ImportError:
            # Skip test if ruamel.yaml is not available
            pytest.skip("ruamel.yaml not available for testing")

    def test_complex_model_toml_serialization(self, tmp_path: Path) -> None:
        """Test TOML serialization of complex nested models."""
        nested = SimpleModel(name="nested_toml", value=2)
        complex_model = ComplexModel(
            items=["toml", "test", "items"],
            mapping={"toml_key": "toml_value", "number": 84},
            nested=nested,
        )
        file_path = tmp_path / "complex.toml"

        try:
            complex_model.to_file(file_path, file_format="toml")
            loaded = ComplexModel.from_file(file_path, file_format="toml")

            assert loaded.items == complex_model.items
            assert loaded.mapping == complex_model.mapping
            assert loaded.nested.name == complex_model.nested.name
            assert loaded.nested.value == complex_model.nested.value
        except ImportError:
            # Skip test if TOML libraries are not available
            pytest.skip("TOML libraries not available for testing")

    def test_format_roundtrip_consistency(self, tmp_path: Path) -> None:
        """Test that all supported formats produce consistent results."""
        original = ComplexModel(
            items=["roundtrip", "test"],
            mapping={"format": "test", "number": 123},
            nested=SimpleModel(name="consistent", value=999),
        )

        json_file = tmp_path / "test.json"
        yaml_file = tmp_path / "test.yaml"
        toml_file = tmp_path / "test.toml"

        # Save in all formats
        original.to_file(json_file, file_format="json")

        try:
            original.to_file(yaml_file, file_format="yaml")
            yaml_loaded = ComplexModel.from_file(yaml_file, file_format="yaml")
            json_loaded = ComplexModel.from_file(json_file, file_format="json")

            # Should be identical
            assert yaml_loaded.items == json_loaded.items
            assert yaml_loaded.mapping == json_loaded.mapping
            assert yaml_loaded.nested.name == json_loaded.nested.name
            assert yaml_loaded.nested.value == json_loaded.nested.value
        except ImportError:
            pass  # Skip YAML part if not available

        try:
            original.to_file(toml_file, file_format="toml")
            toml_loaded = ComplexModel.from_file(toml_file, file_format="toml")
            json_loaded = ComplexModel.from_file(json_file, file_format="json")

            # Should be identical
            assert toml_loaded.items == json_loaded.items
            assert toml_loaded.mapping == json_loaded.mapping
            assert toml_loaded.nested.name == json_loaded.nested.name
            assert toml_loaded.nested.value == json_loaded.nested.value
        except ImportError:
            pass  # Skip TOML part if not available

    def test_auto_format_json_to_file(self, tmp_path: Path) -> None:
        """Test that file_format='auto' correctly detects JSON format from file extension."""
        model = SimpleModel(name="auto_json", value=42)
        file_path = tmp_path / "test.json"

        # Use auto format - should detect JSON from .json extension
        model.to_file(file_path, file_format="auto")

        assert file_path.exists()
        with open(file_path) as f:
            data = json.load(f)
        assert data == {"name": "auto_json", "value": 42}

    def test_auto_format_json_from_file(self, tmp_path: Path) -> None:
        """Test that file_format='auto' correctly loads JSON files."""
        file_path = tmp_path / "test.json"
        data: dict[str, Any] = {"name": "auto_loaded", "value": 123}
        with open(file_path, "w") as f:
            json.dump(data, f)

        # Use auto format - should detect JSON from .json extension
        loaded = SimpleModel.from_file(file_path, file_format="auto")

        assert loaded.name == "auto_loaded"
        assert loaded.value == 123

    def test_auto_format_yaml_to_file(self, tmp_path: Path) -> None:
        """Test that file_format='auto' correctly detects YAML format from file extension."""
        model = SimpleModel(name="auto_yaml", value=456)
        yaml_file = tmp_path / "test.yaml"
        yml_file = tmp_path / "test.yml"

        try:
            # Test .yaml extension
            model.to_file(yaml_file, file_format="auto")
            assert yaml_file.exists()

            # Test .yml extension
            model.to_file(yml_file, file_format="auto")
            assert yml_file.exists()

            # Verify content is valid YAML
            loaded_yaml = SimpleModel.from_file(yaml_file, file_format="auto")
            loaded_yml = SimpleModel.from_file(yml_file, file_format="auto")

            assert loaded_yaml.name == "auto_yaml"
            assert loaded_yaml.value == 456
            assert loaded_yml.name == "auto_yaml"
            assert loaded_yml.value == 456
        except ImportError:
            pytest.skip("ruamel.yaml not available for testing")

    def test_auto_format_yaml_from_file(self, tmp_path: Path) -> None:
        """Test that file_format='auto' correctly loads YAML files."""
        yaml_file = tmp_path / "test.yaml"
        yml_file = tmp_path / "test.yml"

        yaml_content = """
name: auto_yaml_loaded
value: 789
        """.strip()

        try:
            # Test .yaml extension
            yaml_file.write_text(yaml_content)
            loaded_yaml = SimpleModel.from_file(yaml_file, file_format="auto")
            assert loaded_yaml.name == "auto_yaml_loaded"
            assert loaded_yaml.value == 789

            # Test .yml extension
            yml_file.write_text(yaml_content)
            loaded_yml = SimpleModel.from_file(yml_file, file_format="auto")
            assert loaded_yml.name == "auto_yaml_loaded"
            assert loaded_yml.value == 789
        except ImportError:
            pytest.skip("ruamel.yaml not available for testing")

    def test_auto_format_toml_to_file(self, tmp_path: Path) -> None:
        """Test that file_format='auto' correctly detects TOML format from file extension."""
        model = SimpleModel(name="auto_toml", value=789)
        file_path = tmp_path / "test.toml"

        try:
            # Use auto format - should detect TOML from .toml extension
            model.to_file(file_path, file_format="auto")

            assert file_path.exists()

            # Verify content by loading it back
            loaded = SimpleModel.from_file(file_path, file_format="auto")
            assert loaded.name == "auto_toml"
            assert loaded.value == 789
        except ImportError:
            pytest.skip("TOML libraries not available for testing")

    def test_auto_format_toml_from_file(self, tmp_path: Path) -> None:
        """Test that file_format='auto' correctly loads TOML files."""
        file_path = tmp_path / "test.toml"
        toml_content = """
name = "auto_toml_loaded"
value = 999
        """.strip()

        try:
            file_path.write_text(toml_content)
            loaded = SimpleModel.from_file(file_path, file_format="auto")

            assert loaded.name == "auto_toml_loaded"
            assert loaded.value == 999
        except ImportError:
            pytest.skip("TOML libraries not available for testing")

    def test_auto_format_unsupported_extension_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test that file_format='auto' raises ValueError for unsupported extensions."""
        model = SimpleModel(name="test", value=42)
        xml_file = tmp_path / "test.xml"
        txt_file = tmp_path / "test.txt"
        no_ext_file = tmp_path / "test"

        # Test unsupported extensions
        with pytest.raises(ValueError, match="Unsupported file extension: .xml"):
            model.to_file(xml_file, file_format="auto")

        with pytest.raises(ValueError, match="Unsupported file extension: .txt"):
            model.to_file(txt_file, file_format="auto")

        with pytest.raises(ValueError, match="Unsupported file extension: "):
            model.to_file(no_ext_file, file_format="auto")

        # Test from_file with unsupported extensions
        xml_file.write_text('{"name": "test", "value": 42}')
        with pytest.raises(ValueError, match="Unsupported file extension: .xml"):
            SimpleModel.from_file(xml_file, file_format="auto")

    def test_auto_format_complex_model_roundtrip(self, tmp_path: Path) -> None:
        """Test that complex models work correctly with auto format detection."""
        nested = SimpleModel(name="nested_auto", value=42)
        original = ComplexModel(
            items=["auto", "format", "test"],
            mapping={"auto": True, "format": "detected", "value": 123},
            nested=nested,
        )

        json_file = tmp_path / "complex_auto.json"
        yaml_file = tmp_path / "complex_auto.yaml"
        toml_file = tmp_path / "complex_auto.toml"

        # Test JSON auto-detection
        original.to_file(json_file, file_format="auto")
        json_loaded = ComplexModel.from_file(json_file, file_format="auto")
        assert json_loaded == original

        try:
            # Test YAML auto-detection
            original.to_file(yaml_file, file_format="auto")
            yaml_loaded = ComplexModel.from_file(yaml_file, file_format="auto")
            assert yaml_loaded == original
        except ImportError:
            pass  # Skip YAML part if not available

        try:
            # Test TOML auto-detection
            original.to_file(toml_file, file_format="auto")
            toml_loaded = ComplexModel.from_file(toml_file, file_format="auto")
            assert toml_loaded == original
        except ImportError:
            pass  # Skip TOML part if not available

    def test_auto_format_is_default_behavior(self, tmp_path: Path) -> None:
        """Test that file_format='auto' is the default behavior when not specified."""
        model = SimpleModel(name="default_auto", value=42)
        json_file = tmp_path / "default.json"
        yaml_file = tmp_path / "default.yaml"

        # Don't specify file_format - should default to "auto"
        model.to_file(json_file)
        loaded_json = SimpleModel.from_file(json_file)

        assert loaded_json.name == "default_auto"
        assert loaded_json.value == 42

        try:
            model.to_file(yaml_file)
            loaded_yaml = SimpleModel.from_file(yaml_file)

            assert loaded_yaml.name == "default_auto"
            assert loaded_yaml.value == 42
        except ImportError:
            pass  # Skip YAML part if not available

    def test_auto_format_case_insensitive_extensions(self, tmp_path: Path) -> None:
        """Test that auto format detection is case-insensitive for file extensions."""
        model = SimpleModel(name="case_test", value=42)

        # Test various case combinations
        json_upper = tmp_path / "test.JSON"
        yaml_upper = tmp_path / "test.YAML"
        yml_mixed = tmp_path / "test.YmL"
        toml_upper = tmp_path / "test.TOML"

        # JSON - should work regardless of case
        model.to_file(json_upper, file_format="auto")
        loaded = SimpleModel.from_file(json_upper, file_format="auto")
        assert loaded.name == "case_test"

        try:
            # YAML extensions - should work regardless of case
            model.to_file(yaml_upper, file_format="auto")
            loaded = SimpleModel.from_file(yaml_upper, file_format="auto")
            assert loaded.name == "case_test"

            model.to_file(yml_mixed, file_format="auto")
            loaded = SimpleModel.from_file(yml_mixed, file_format="auto")
            assert loaded.name == "case_test"
        except ImportError:
            pass  # Skip YAML part if not available

        try:
            # TOML - should work regardless of case
            model.to_file(toml_upper, file_format="auto")
            loaded = SimpleModel.from_file(toml_upper, file_format="auto")
            assert loaded.name == "case_test"
        except ImportError:
            pass  # Skip TOML part if not available


# Helper Function Tests
class TestHelperFunctions:
    def test_file_format_from_file_json(self) -> None:
        """Test file_format_from_file correctly detects JSON files."""
        from pathlib import Path

        from pydantic_store.models import file_format_from_file

        assert file_format_from_file(Path("test.json")) == "json"
        assert file_format_from_file(Path("data.JSON")) == "json"  # Case insensitive
        assert file_format_from_file(Path("/path/to/file.json")) == "json"

    def test_file_format_from_file_yaml(self) -> None:
        """Test file_format_from_file correctly detects YAML files."""
        from pathlib import Path

        from pydantic_store.models import file_format_from_file

        assert file_format_from_file(Path("test.yaml")) == "yaml"
        assert file_format_from_file(Path("test.yml")) == "yaml"
        assert file_format_from_file(Path("config.YAML")) == "yaml"  # Case insensitive
        assert file_format_from_file(Path("config.YML")) == "yaml"  # Case insensitive

    def test_file_format_from_file_toml(self) -> None:
        """Test file_format_from_file correctly detects TOML files."""
        from pathlib import Path

        from pydantic_store.models import file_format_from_file

        assert file_format_from_file(Path("pyproject.toml")) == "toml"
        assert file_format_from_file(Path("config.TOML")) == "toml"  # Case insensitive

    def test_file_format_from_file_unsupported_extension(self) -> None:
        """Test file_format_from_file raises ValueError for unsupported extensions."""
        from pathlib import Path

        from pydantic_store.models import file_format_from_file

        with pytest.raises(ValueError, match="Unsupported file extension: .xml"):
            file_format_from_file(Path("test.xml"))

        with pytest.raises(ValueError, match="Unsupported file extension: .txt"):
            file_format_from_file(Path("test.txt"))

        with pytest.raises(ValueError, match="Unsupported file extension: .py"):
            file_format_from_file(Path("script.py"))

        # File with no extension
        with pytest.raises(ValueError, match="Unsupported file extension: "):
            file_format_from_file(Path("noextension"))

    def test_yaml_to_json_conversion(self) -> None:
        """Test yaml_to_json helper function."""
        yaml_string = """
name: test_yaml
value: 123
items:
  - item1
  - item2
mapping:
  key1: value1
  key2: 42
        """.strip()

        try:
            json_string = yaml_to_json(yaml_string)
            data = json.loads(json_string)

            assert data["name"] == "test_yaml"
            assert data["value"] == 123
            assert data["items"] == ["item1", "item2"]
            assert data["mapping"] == {"key1": "value1", "key2": 42}
        except ImportError:
            pytest.skip("ruamel.yaml not available for testing")

    def test_toml_to_json_conversion(self) -> None:
        """Test toml_to_json helper function."""
        toml_string = """
name = "test_toml"
value = 456

[mapping]
key1 = "value1"
key2 = 84

[[items]]
name = "item1"

[[items]]
name = "item2"
        """.strip()

        try:
            json_string = toml_to_json(toml_string)
            data = json.loads(json_string)

            assert data["name"] == "test_toml"
            assert data["value"] == 456
            assert data["mapping"] == {"key1": "value1", "key2": 84}
            assert len(data["items"]) == 2
            assert data["items"][0]["name"] == "item1"
            assert data["items"][1]["name"] == "item2"
        except ImportError:
            pytest.skip("TOML libraries not available for testing")

    def test_yaml_to_json_error_handling(self) -> None:
        """Test that yaml_to_json raises ImportError when ruamel.yaml is not available."""
        # This test will pass if ruamel.yaml is available, but validates the error message structure
        try:
            yaml_to_json("name: test")
        except ImportError as e:
            assert "ruamel.yaml is required for YAML support" in str(e)
            assert "pip install ruamel.yaml" in str(e)

    def test_toml_to_json_error_handling(self) -> None:
        """Test that toml_to_json raises ImportError when TOML libraries are not available."""
        # This test will pass if TOML libraries are available, but validates the error message structure
        try:
            toml_to_json('name = "test"')
        except ImportError as e:
            assert "tomllib" in str(
                e
            ) or "tomli is required for TOML reading support" in str(e)

    def test_toml_root_array_validation(self, tmp_path: Path) -> None:
        """Test that TOML format properly validates against root-level arrays."""
        # This should work fine - BaseModel produces an object at root
        model = SimpleModel(name="toml_test", value=42)
        toml_file = tmp_path / "valid.toml"

        try:
            model.to_file(toml_file, file_format="toml")
            # Should succeed - BaseModel creates a dict/object
            assert toml_file.exists()
        except ImportError:
            pytest.skip("TOML libraries not available for testing")


# ListModel Tests
class TestListModelClass:
    def test_list_initialization(self) -> None:
        """Test ListModel can be initialized empty or with data."""
        empty_list = StringListModel()
        assert len(empty_list) == 0
        assert list(empty_list) == []

        initialized_list = StringListModel(root=["a", "b", "c"])
        assert len(initialized_list) == 3
        assert list(initialized_list) == ["a", "b", "c"]

    def test_append_functionality(self) -> None:
        """Test the append method works like a regular list."""
        model = StringListModel()
        model.append("first")
        model.append("second")

        assert len(model) == 2
        assert model[0] == "first"
        assert model[1] == "second"

    def test_extend_functionality(self) -> None:
        """Test the extend method works like a regular list."""
        model = StringListModel(root=["a"])
        model.extend(["b", "c", "d"])

        assert len(model) == 4
        assert list(model) == ["a", "b", "c", "d"]

    def test_indexing_access(self) -> None:
        """Test getting items by index."""
        model = StringListModel(root=["zero", "one", "two", "three"])

        assert model[0] == "zero"
        assert model[1] == "one"
        assert model[-1] == "three"
        assert model[-2] == "two"

    def test_slice_access(self) -> None:
        """Test getting items by slice."""
        model = StringListModel(root=["a", "b", "c", "d", "e"])

        assert model[1:3] == ["b", "c"]
        assert model[:2] == ["a", "b"]
        assert model[2:] == ["c", "d", "e"]
        assert model[::2] == ["a", "c", "e"]

    def test_item_assignment(self) -> None:
        """Test setting items by index."""
        model = StringListModel(root=["a", "b", "c"])
        model[1] = "modified"

        assert model[1] == "modified"
        assert list(model) == ["a", "modified", "c"]

    def test_slice_assignment(self) -> None:
        """Test setting items by slice."""
        model = StringListModel(root=["a", "b", "c", "d"])
        model[1:3] = ["x", "y"]

        assert list(model) == ["a", "x", "y", "d"]

    def test_len_functionality(self) -> None:
        """Test len() works correctly."""
        model = StringListModel()
        assert len(model) == 0

        model.append("item")
        assert len(model) == 1

        model.extend(["a", "b", "c"])
        assert len(model) == 4

    def test_iteration(self) -> None:
        """Test iteration over the list model."""
        items = ["first", "second", "third"]
        model = StringListModel(root=items)

        result: list[str] = []
        for item in model:
            result.append(item)

        assert result == items

    def test_contains_functionality(self):
        """Test 'in' operator works correctly."""
        model = StringListModel(root=["apple", "banana", "cherry"])

        assert "apple" in model
        assert "banana" in model
        assert "grape" not in model

    def test_list_like_behavior_comprehensive(self) -> None:
        """Test that ListModel behaves exactly like a regular list."""
        regular_list = ["a", "b", "c"]
        model = StringListModel(root=["a", "b", "c"])

        # Test equivalence operations
        assert len(model) == len(regular_list)
        assert list(model) == regular_list

        # Test modifications
        regular_list.append("d")
        model.append("d")
        assert list(model) == regular_list

        regular_list.extend(["e", "f"])
        model.extend(["e", "f"])
        assert list(model) == regular_list

    def test_empty_list_behavior(self) -> None:
        """Test edge cases with empty lists."""
        model = StringListModel()

        assert len(model) == 0
        assert list(model) == []
        assert "anything" not in model

        # Test that iteration over empty list works
        items = list(iter(model))
        assert items == []

    def test_serialization_roundtrip(self, tmp_path: Path) -> None:
        """Test that ListModel can be serialized and deserialized."""
        original = StringListModel(["serialize", "me", "please"])
        file_path = tmp_path / "list_model.json"

        original.to_file(file_path)
        loaded = StringListModel.from_file(file_path)

        assert list(loaded) == list(original)
        assert len(loaded) == len(original)

    def test_auto_format_serialization(self, tmp_path: Path) -> None:
        """Test that ListModel works with auto format detection."""
        original = StringListModel(["auto", "format", "list", "test"])

        # Test with different extensions
        json_file = tmp_path / "list_auto.json"
        yaml_file = tmp_path / "list_auto.yaml"
        toml_file = tmp_path / "list_auto.toml"

        # JSON auto-detection
        original.to_file(json_file, file_format="auto")
        loaded_json = StringListModel.from_file(json_file, file_format="auto")
        assert list(loaded_json) == list(original)

        try:
            # YAML auto-detection
            original.to_file(yaml_file, file_format="auto")
            loaded_yaml = StringListModel.from_file(yaml_file, file_format="auto")
            assert list(loaded_yaml) == list(original)
        except ImportError:
            pass  # Skip YAML part if not available

        try:
            # TOML auto-detection should fail for lists at root level
            with pytest.raises(
                ValueError,
                match="TOML format does not support arrays at the root level",
            ):
                original.to_file(toml_file, file_format="auto")
        except ImportError:
            pass  # Skip TOML part if not available


# DictModel Tests
class TestDictModelClass:
    def test_dict_initialization(self) -> None:
        """Test DictModel can be initialized empty or with data."""
        empty_dict = StringIntDictModel()
        assert len(empty_dict) == 0
        assert dict(empty_dict) == {}

        initialized_dict = StringIntDictModel(root={"a": 1, "b": 2})
        assert len(initialized_dict) == 2
        assert dict(initialized_dict) == {"a": 1, "b": 2}

    def test_item_access(self) -> None:
        """Test getting and setting items."""
        model = StringIntDictModel(root={"key1": 10, "key2": 20})

        assert model["key1"] == 10
        assert model["key2"] == 20

        model["key3"] = 30
        assert model["key3"] == 30

    def test_contains_functionality(self) -> None:
        """Test 'in' operator works correctly."""
        model = StringIntDictModel(root={"apple": 1, "banana": 2})

        assert "apple" in model
        assert "banana" in model
        assert "grape" not in model

    def test_keys_values_items(self) -> None:
        """Test dict-like methods for accessing keys, values, and items."""
        data = {"a": 1, "b": 2, "c": 3}
        model = StringIntDictModel(root=data)

        assert set(model.keys()) == set(data.keys())
        assert set(model.values()) == set(data.values())
        assert set(model.items()) == set(data.items())

    def test_dict_modification(self) -> None:
        """Test modifying the dictionary."""
        model = StringIntDictModel()

        model["first"] = 1
        model["second"] = 2

        assert model["first"] == 1
        assert model["second"] == 2
        assert len(model) == 2

    def test_dict_like_behavior_comprehensive(self) -> None:
        """Test that DictModel behaves exactly like a regular dict."""
        regular_dict = {"a": 1, "b": 2, "c": 3}
        model = StringIntDictModel(root={"a": 1, "b": 2, "c": 3})

        # Test equivalence operations
        assert len(model) == len(regular_dict)
        assert dict(model) == regular_dict
        assert set(model.keys()) == set(regular_dict.keys())
        assert set(model.values()) == set(regular_dict.values())

        # Test modifications
        regular_dict["d"] = 4
        model["d"] = 4
        assert dict(model) == regular_dict

    def test_serialization_roundtrip(self, tmp_path: Path) -> None:
        """Test that DictModel can be serialized and deserialized."""
        original = StringIntDictModel({"serialize": 1, "me": 2, "please": 3})
        file_path = tmp_path / "dict_model.json"

        original.to_file(file_path)
        loaded = StringIntDictModel.from_file(file_path)

        assert dict(loaded) == dict(original)
        assert len(loaded) == len(original)

    def test_auto_format_serialization(self, tmp_path: Path) -> None:
        """Test that DictModel works with auto format detection."""
        original = StringIntDictModel({"auto": 1, "format": 2, "dict": 3, "test": 4})

        # Test with different extensions
        json_file = tmp_path / "dict_auto.json"
        yaml_file = tmp_path / "dict_auto.yaml"
        toml_file = tmp_path / "dict_auto.toml"

        # JSON auto-detection
        original.to_file(json_file, file_format="auto")
        loaded_json = StringIntDictModel.from_file(json_file, file_format="auto")
        assert dict(loaded_json) == dict(original)

        try:
            # YAML auto-detection
            original.to_file(yaml_file, file_format="auto")
            loaded_yaml = StringIntDictModel.from_file(yaml_file, file_format="auto")
            assert dict(loaded_yaml) == dict(original)
        except ImportError:
            pass  # Skip YAML part if not available

        try:
            # TOML auto-detection
            original.to_file(toml_file, file_format="auto")
            loaded_toml = StringIntDictModel.from_file(toml_file, file_format="auto")
            assert dict(loaded_toml) == dict(original)
        except ImportError:
            pass  # Skip TOML part if not available


# JsonStore Tests
class TestJsonStoreClass:
    def test_initialization_without_file(self) -> None:
        """Test JsonStore initialization without file path."""
        store = JsonStore[int]()
        assert len(store) == 0
        assert dict(store) == {}

    def test_initialization_with_data(self) -> None:
        """Test JsonStore initialization with data."""
        data = {"key1": "value1", "key2": "value2"}
        store = JsonStore[str](root=data)
        assert dict(store) == data

    def test_connect_creates_new_file(self, tmp_path: Path) -> None:
        """Test that connect creates a new file if it doesn't exist."""
        file_path = tmp_path / "new_store.json"
        assert not file_path.exists()

        store = JsonStore[str].connect(file_path)

        assert file_path.exists()
        assert len(store) == 0
        assert dict(store) == {}

    def test_connect_loads_existing_file(self, tmp_path: Path) -> None:
        """Test that connect loads an existing file."""
        file_path = tmp_path / "existing_store.json"
        data: dict[str, Any] = {"existing": "data", "count": 42}

        # Create the file manually
        with open(file_path, "w") as f:
            json.dump(data, f)

        store = JsonStore[Union[str, int]].connect(file_path)

        assert dict(store) == data
        assert store["existing"] == "data"
        assert store["count"] == 42

    def test_auto_save_on_setitem(self, tmp_path: Path) -> None:
        """Test that setting items automatically saves to file."""
        file_path = tmp_path / "auto_save.json"
        store = JsonStore[str].connect(file_path)

        store["key"] = "value"

        # Verify it was saved to file immediately
        with open(file_path) as f:
            saved_data = json.load(f)
        assert saved_data == {"key": "value"}

    def test_multiple_operations_persist(self, tmp_path: Path) -> None:
        """Test that multiple operations all persist to file."""
        file_path = tmp_path / "multi_ops.json"
        store = JsonStore[str].connect(file_path)

        store["first"] = "value1"
        store["second"] = "value2"
        store["third"] = "value3"

        # Create a new store instance to verify persistence
        new_store = JsonStore[str].connect(file_path)
        assert new_store["first"] == "value1"
        assert new_store["second"] == "value2"
        assert new_store["third"] == "value3"

    def test_save_store_without_path_raises_error(self) -> None:
        """Test that save_store raises error when no file path is set."""
        store = JsonStore[str]()

        with pytest.raises(ValueError, match="File path is not set"):
            store.save_store()

    def test_manual_save_store(self, tmp_path: Path) -> None:
        """Test manual save_store functionality."""
        file_path = tmp_path / "manual_save.json"
        store = JsonStore[str](root={"initial": "data"}, file_path=file_path)

        # Manual save
        store.save_store()

        assert file_path.exists()
        with open(file_path) as f:
            data = json.load(f)
        assert data == {"initial": "data"}

    def test_file_path_persistence(self, tmp_path: Path) -> None:
        """Test that file path is maintained across operations."""
        file_path = tmp_path / "path_persist.json"
        store = JsonStore[str].connect(file_path)

        # Verify the file path is stored
        assert store._file_path == file_path  # type: ignore

        # Operations should still work
        store["test"] = "works"
        assert file_path.exists()

    def test_jsonstore_with_auto_format(self, tmp_path: Path) -> None:
        """Test that JsonStore works with auto format detection."""
        # Test different file extensions
        json_file = tmp_path / "store.json"
        yaml_file = tmp_path / "store.yaml"
        toml_file = tmp_path / "store.toml"

        original_data = {"key1": "value1", "key2": "value2"}

        # Test JSON auto-detection
        store = JsonStore[str](root=original_data, file_path=json_file)
        store.to_file(json_file, file_format="auto")
        loaded = JsonStore[str].from_file(json_file, file_format="auto")
        assert dict(loaded) == original_data

        try:
            # Test YAML auto-detection
            store.to_file(yaml_file, file_format="auto")
            loaded = JsonStore[str].from_file(yaml_file, file_format="auto")
            assert dict(loaded) == original_data
        except ImportError:
            pass  # Skip YAML part if not available

        try:
            # Test TOML auto-detection
            store.to_file(toml_file, file_format="auto")
            loaded = JsonStore[str].from_file(toml_file, file_format="auto")
            assert dict(loaded) == original_data
        except ImportError:
            pass  # Skip TOML part if not available
