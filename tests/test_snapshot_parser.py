"""Tests for the GhostRef snapshot parser."""

from tests.conftest import _test_getattr

from pydantic import JsonValue

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ml_switcheroo.ingestion.snapshot_parser import (
  FrameworkSnapshot,
  SnapshotArgument,
  SnapshotOperation,
  parse_snapshot_file,
  parse_snapshot_string,
)


@pytest.fixture
def valid_snapshot_data() -> dict[str, JsonValue]:
  """Fixture for valid snapshot JSON data.

  Returns:
      Configured `valid_snapshot_data` fixture value (dict[str, JsonValue]).
  """
  return {
    "framework_name": "pytorch",
    "version": "2.0.0",
    "operations": [
      {
        "name": "torch.nn.Linear",
        "kind": "class",
        "arguments": [
          {"name": "in_features", "type_hint": "int", "default": None, "description": "size of each input sample"},
          {"name": "out_features", "type_hint": "int", "default": None, "description": "size of each output sample"},
          {
            "name": "bias",
            "type_hint": "bool",
            "default": "True",
            "description": "If set to False, the layer will not learn an additive bias.",
          },
        ],
        "returns": "Linear",
        "docstring": "Applies a linear transformation to the incoming data.",
        "is_overload": False,
      }
    ],
    "metadata": {"cuda_available": True},
  }


def test_snapshot_argument_model() -> None:
  """Test SnapshotArgument validation."""
  arg = SnapshotArgument(name="x", type_hint="float", default="0.0")
  assert arg.name == "x"
  assert arg.type_hint == "float"
  assert arg.default == "0.0"

  with pytest.raises(ValidationError):
    SnapshotArgument.model_validate({"type_hint": "float"})  # missing required 'name'


def test_snapshot_operation_model() -> None:
  """Test SnapshotOperation validation."""
  op = SnapshotOperation(name="my_op", kind="function")
  assert op.name == "my_op"
  assert op.kind == "function"
  assert op.arguments == []
  assert op.is_overload is False


def test_framework_snapshot_model(valid_snapshot_data: dict[str, JsonValue]) -> None:
  """Test FrameworkSnapshot validation with full data.

  Args:
      valid_snapshot_data: The valid snapshot data (dict[str, JsonValue]) for test framework snapshot model.
  """
  snapshot = _test_getattr(
    {
      "v": _test_getattr({"v": FrameworkSnapshot}, "__getitem__")("v")(
        **_test_getattr({"v": valid_snapshot_data}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert snapshot.framework_name == "pytorch"
  assert len(snapshot.operations) == 1
  assert snapshot.operations[0].name == "torch.nn.Linear"
  assert snapshot.metadata == {"cuda_available": True}


def test_parse_snapshot_string_valid(valid_snapshot_data: dict[str, JsonValue]) -> None:
  """Test parsing a valid JSON string.

  Args:
      valid_snapshot_data: The valid snapshot data (dict[str, JsonValue]) for test parse snapshot string valid.
  """
  json_str = json.dumps(valid_snapshot_data)
  snapshot = parse_snapshot_string(json_str)
  assert snapshot.framework_name == "pytorch"


def test_parse_snapshot_string_invalid_json() -> None:
  """Test parsing an invalid JSON string."""
  with pytest.raises(ValueError, match="Failed to decode JSON string"):
    parse_snapshot_string("{invalid_json: true,")


def test_parse_snapshot_string_invalid_schema() -> None:
  """Test parsing a string that fails schema validation."""
  invalid_data = {"framework_name": "pytorch"}  # Missing version
  with pytest.raises(ValueError, match="Snapshot data does not match expected schema"):
    parse_snapshot_string(json.dumps(invalid_data))


def test_parse_snapshot_file_valid(tmp_path: Path, valid_snapshot_data: dict[str, JsonValue]) -> None:
  """Test parsing a valid JSON file.

  Args:
      tmp_path: Pytest temporary directory path fixture.
      valid_snapshot_data: The valid snapshot data (dict[str, JsonValue]) for test parse snapshot file valid.
  """
  file_path = tmp_path / "snapshot.json"
  file_path.write_text(json.dumps(valid_snapshot_data), encoding="utf-8")
  snapshot = parse_snapshot_file(file_path)
  assert snapshot.framework_name == "pytorch"


def test_parse_snapshot_file_not_found(tmp_path: Path) -> None:
  """Test parsing a non-existent file.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  file_path = tmp_path / "does_not_exist.json"
  with pytest.raises(FileNotFoundError, match="Snapshot file not found"):
    parse_snapshot_file(file_path)


def test_parse_snapshot_file_invalid_json(tmp_path: Path) -> None:
  """Test parsing a file with invalid JSON.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  file_path = tmp_path / "bad.json"
  file_path.write_text("{bad: [}", encoding="utf-8")
  with pytest.raises(ValueError, match="Failed to decode JSON from"):
    parse_snapshot_file(file_path)


def test_parse_snapshot_file_invalid_schema(tmp_path: Path) -> None:
  """Test parsing a file with invalid schema.

  Args:
      tmp_path: Pytest temporary directory path fixture.
  """
  file_path = tmp_path / "invalid.json"
  file_path.write_text('{"framework_name": "pytorch"}', encoding="utf-8")
  with pytest.raises(ValueError, match="Snapshot data does not match expected schema"):
    parse_snapshot_file(file_path)


def test_parse_v006_envelope_and_hierarchy() -> None:
  """Test parsing a v0.0.6 SnapshotEnvelope with categories, classes, fields, functions, args, and docstrings."""
  envelope_data: dict[str, JsonValue] = {
    "target": "torch",
    "version": "2.15.0+cpu",
    "schema_version": "2.0.0",
    "generated_at": "2026-11-10T00:00:00Z",
    "categories": {
      "layer": [
        {
          "name": "Linear",
          "api_path": "torch.nn.Linear",
          "kind": "class",
          "docstring": "Applies an affine linear transformation.",
          "returns_type": "NoneType",
          "params": [
            {
              "name": "in_features",
              "annotation": "int",
              "default": None,
              "description": "size of each input sample",
              "kind": "POSITIONAL_OR_KEYWORD",
              "is_mandatory": True,
            },
            {
              "name": "bias",
              "annotation": "bool",
              "default": True,
              "description": "If set to False, the layer will not learn an additive bias.",
              "kind": "POSITIONAL_OR_KEYWORD",
              "is_mandatory": False,
            },
          ],
          "overloads": ["(in_features: int, out_features: int, bias: bool = True)"],
        }
      ],
      "array": [
        {
          "name": "top_level_fn",
          "api_path": "",
          "kind": "function",
          "docstring": "Top-level function without dot in name.",
          "returns_type": "Tensor",
          "params": [
            {
              "name": "x",
              "type_hint": "Tensor",
              "default": "0",
              "description": "Input tensor",
            }
          ],
        }
      ],
      "non_list_ignored": "ignored",
    },
  }
  snapshot = parse_snapshot_string(json.dumps(envelope_data))
  assert snapshot.framework_name == "torch"
  assert snapshot.version == "2.15.0+cpu"
  assert len(snapshot.operations) == 2

  classes = snapshot.classes()
  assert len(classes) == 1
  assert classes[0].name == "Linear"
  assert classes[0].api_path == "torch.nn.Linear"
  assert classes[0].returns == "NoneType"
  assert classes[0].is_overload is True
  assert len(classes[0].fields) == 2
  assert classes[0].fields[0].type_hint == "int"
  assert classes[0].fields[0].annotation == "int"
  assert classes[0].fields[1].default == "True"

  funcs = snapshot.functions()
  assert len(funcs) == 1
  assert funcs[0].name == "top_level_fn"
  assert funcs[0].api_path == "top_level_fn"
  assert funcs[0].arguments[0].annotation == "Tensor"

  hierarchy = snapshot.module_hierarchy()
  assert "torch.nn" in hierarchy
  assert "torch" in hierarchy

  # Branch where kind=='class' and fields is explicitly provided, or api_path is empty on op
  op_direct = SnapshotOperation(name="BareOp", api_path="", kind="class")
  snap_direct = FrameworkSnapshot(framework_name="custom", version="1.0", operations=[op_direct])
  snap_direct.operations[0].api_path = ""
  assert "custom" in snap_direct.module_hierarchy()
  raw_cls_with_fields = SnapshotOperation.model_validate({"name": "ClsWithFields", "kind": "class", "fields": []})
  assert raw_cls_with_fields.fields == []
