"""Tests for the GhostRef snapshot parser."""

import json
import pytest
from pathlib import Path
from pydantic import ValidationError

from ml_switcheroo.ingestion.snapshot_parser import (
  SnapshotArgument,
  SnapshotOperation,
  FrameworkSnapshot,
  parse_snapshot_file,
  parse_snapshot_string,
)


@pytest.fixture
def valid_snapshot_data():
  """Fixture for valid snapshot JSON data."""
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


def test_snapshot_argument_model():
  """Test SnapshotArgument validation."""
  arg = SnapshotArgument(name="x", type_hint="float", default="0.0")
  assert arg.name == "x"
  assert arg.type_hint == "float"
  assert arg.default == "0.0"

  with pytest.raises(ValidationError):
    SnapshotArgument(type_hint="float")  # missing required 'name'


def test_snapshot_operation_model():
  """Test SnapshotOperation validation."""
  op = SnapshotOperation(name="my_op", kind="function")
  assert op.name == "my_op"
  assert op.kind == "function"
  assert op.arguments == []
  assert op.is_overload is False


def test_framework_snapshot_model(valid_snapshot_data):
  """Test FrameworkSnapshot validation with full data."""
  snapshot = FrameworkSnapshot(**valid_snapshot_data)
  assert snapshot.framework_name == "pytorch"
  assert len(snapshot.operations) == 1
  assert snapshot.operations[0].name == "torch.nn.Linear"
  assert snapshot.metadata == {"cuda_available": True}


def test_parse_snapshot_string_valid(valid_snapshot_data):
  """Test parsing a valid JSON string."""
  json_str = json.dumps(valid_snapshot_data)
  snapshot = parse_snapshot_string(json_str)
  assert snapshot.framework_name == "pytorch"


def test_parse_snapshot_string_invalid_json():
  """Test parsing an invalid JSON string."""
  with pytest.raises(ValueError, match="Failed to decode JSON string"):
    parse_snapshot_string("{invalid_json: true,")


def test_parse_snapshot_string_invalid_schema():
  """Test parsing a string that fails schema validation."""
  invalid_data = {"framework_name": "pytorch"}  # Missing version
  with pytest.raises(ValueError, match="Snapshot data does not match expected schema"):
    parse_snapshot_string(json.dumps(invalid_data))


def test_parse_snapshot_file_valid(tmp_path: Path, valid_snapshot_data):
  """Test parsing a valid JSON file."""
  file_path = tmp_path / "snapshot.json"
  file_path.write_text(json.dumps(valid_snapshot_data), encoding="utf-8")
  snapshot = parse_snapshot_file(file_path)
  assert snapshot.framework_name == "pytorch"


def test_parse_snapshot_file_not_found(tmp_path: Path):
  """Test parsing a non-existent file."""
  file_path = tmp_path / "does_not_exist.json"
  with pytest.raises(FileNotFoundError, match="Snapshot file not found"):
    parse_snapshot_file(file_path)


def test_parse_snapshot_file_invalid_json(tmp_path: Path):
  """Test parsing a file with invalid JSON."""
  file_path = tmp_path / "bad.json"
  file_path.write_text("{bad: [}", encoding="utf-8")
  with pytest.raises(ValueError, match="Failed to decode JSON from"):
    parse_snapshot_file(file_path)


def test_parse_snapshot_file_invalid_schema(tmp_path: Path):
  """Test parsing a file with invalid schema."""
  file_path = tmp_path / "invalid.json"
  file_path.write_text('{"framework_name": "pytorch"}', encoding="utf-8")
  with pytest.raises(ValueError, match="Snapshot data does not match expected schema"):
    parse_snapshot_file(file_path)
