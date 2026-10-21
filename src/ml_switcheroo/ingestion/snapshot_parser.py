"""Parser for ml-ecosystem-snapshots Ghost Protocol v2 JSON formats.

This module provides Pydantic models to strictly define the schema of
`ml-ecosystem-snapshots` Ghost Protocol v2 GhostRefs and functions to load them into memory.
"""

import json
from pathlib import Path

from pydantic import BaseModel, Field, JsonValue, ValidationError


class SnapshotArgument(BaseModel):
  """Represents a single argument in a GhostRef operation."""

  name: str = Field(..., description="Name of the argument.")
  type_hint: str | None = Field(None, description="String representation of the type hint.")
  default: str | None = Field(None, description="String representation of the default value, if any.")
  description: str | None = Field(None, description="Docstring description for this argument.")


class SnapshotOperation(BaseModel):
  """Represents a single API operation (function or class) in a framework snapshot."""

  name: str = Field(..., description="Fully qualified name of the operation (e.g., torch.nn.Linear).")
  kind: str = Field(..., description="Type of operation, usually 'function' or 'class'.")
  arguments: list[SnapshotArgument] = Field(default_factory=list, description="List of arguments.")
  returns: str | None = Field(None, description="String representation of the return type.")
  docstring: str | None = Field(None, description="Full docstring of the operation.")
  is_overload: bool = Field(False, description="Whether this is an overload of another function.")


class FrameworkSnapshot(BaseModel):
  """Represents a complete GhostRef snapshot of a machine learning framework."""

  framework_name: str = Field(..., description="Name of the framework (e.g., 'pytorch', 'jax').")
  version: str = Field(..., description="Version of the framework at the time of the snapshot.")
  operations: list[SnapshotOperation] = Field(default_factory=list, description="List of operations in the framework.")
  metadata: dict[str, JsonValue] = Field(default_factory=dict, description="Additional environment metadata.")


def parse_snapshot_file(file_path: str | Path) -> FrameworkSnapshot:
  """Parses a GhostRef snapshot JSON file into memory.

  Args:
      file_path: The path to the JSON snapshot file.

  Returns:
      A strongly-typed FrameworkSnapshot object.

  Raises:
      FileNotFoundError: If the file does not exist.
      ValueError: If the file is not valid JSON or does not match the schema.
  """
  path = Path(file_path)
  if not path.is_file():
    raise FileNotFoundError(f"Snapshot file not found: {path}")

  try:
    with open(path, "r", encoding="utf-8") as f:
      data = json.load(f)
  except json.JSONDecodeError as e:
    raise ValueError(f"Failed to decode JSON from {path}: {e}")

  try:
    return FrameworkSnapshot(**data)
  except ValidationError as e:
    raise ValueError(f"Snapshot data does not match expected schema: {e}")


def parse_snapshot_string(json_data: str) -> FrameworkSnapshot:
  """Parses a GhostRef snapshot JSON string into memory.

  Args:
      json_data: The JSON string representing the snapshot.

  Returns:
      A strongly-typed FrameworkSnapshot object.

  Raises:
      ValueError: If the string is not valid JSON or does not match the schema.
  """
  try:
    data = json.loads(json_data)
  except json.JSONDecodeError as e:
    raise ValueError(f"Failed to decode JSON string: {e}")

  try:
    return FrameworkSnapshot(**data)
  except ValidationError as e:
    raise ValueError(f"Snapshot data does not match expected schema: {e}")
