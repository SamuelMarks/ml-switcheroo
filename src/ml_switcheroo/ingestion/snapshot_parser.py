"""Parser for ml-ecosystem-snapshots Ghost Protocol v2 JSON formats.

This module provides Pydantic models to strictly define the schema of
`ml-ecosystem-snapshots` Ghost Protocol v2 GhostRefs and v0.0.6 SnapshotEnvelopes
(including module hierarchy, classes, fields, functions, function arguments with
types and defaults, and docstrings) and functions to load them into memory.
"""

import json
import typing
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator


class SnapshotArgument(BaseModel):
  """Represents a single argument or class field in a GhostRef operation."""

  model_config = ConfigDict(extra="ignore")

  name: str = Field(..., description="Name of the argument.")
  type_hint: str | None = Field(None, description="String representation of the type hint.")
  annotation: str | None = Field(None, description="Raw type annotation from snapshot inspection.")
  default: str | None = Field(None, description="String representation of the default value, if any.")
  default_factory: str | None = Field(None, description="Default factory expression if applicable.")
  description: str | None = Field(None, description="Docstring description for this argument.")
  kind: str = Field("POSITIONAL_OR_KEYWORD", description="Parameter kind classification.")
  is_mandatory: bool = Field(False, description="Whether the argument is mandatory.")
  allowed_dtypes: list[str] | None = Field(None, description="Permitted tensor dtypes if constrained.")
  allowed_values: list[JsonValue] | None = Field(None, description="Permitted literal values if constrained.")
  rank_constraint: str | int | None = Field(None, description="Tensor rank constraint if specified.")
  standardized_name: str | None = Field(None, description="Normalized ODL standard argument name.")

  @model_validator(mode="before")
  @classmethod
  def _normalize_argument_fields(
    cls,
    data: dict[str, JsonValue] | typing.Mapping[str, JsonValue],
  ) -> dict[str, JsonValue]:
    """Synchronize `annotation` and `type_hint` and normalize scalar defaults to strings.

    Args:
        data: Raw dictionary of argument properties.

    Returns:
        Normalized argument dictionary.
    """
    out: dict[str, JsonValue] = dict(data)
    if out.get("annotation") is not None and out.get("type_hint") is None:
      out["type_hint"] = str(out["annotation"])
    if out.get("type_hint") is not None and out.get("annotation") is None:
      out["annotation"] = str(out["type_hint"])
    raw_default = out.get("default")
    if raw_default is not None and not isinstance(raw_default, str):
      out["default"] = str(raw_default)
    return out


class SnapshotOperation(BaseModel):
  """Represents a single API operation (function or class) in a framework snapshot."""

  model_config = ConfigDict(extra="ignore")

  name: str = Field(..., description="Fully qualified name of the operation (e.g., torch.nn.Linear).")
  api_path: str = Field("", description="Fully qualified import path of the symbol.")
  kind: str = Field(..., description="Type of operation, usually 'function' or 'class'.")
  arguments: list[SnapshotArgument] = Field(default_factory=list, description="List of arguments.")
  fields: list[SnapshotArgument] = Field(
    default_factory=list, description="List of fields or constructor parameters for class definitions."
  )
  returns: str | None = Field(None, description="String representation of the return type.")
  returns_description: str | None = Field(None, description="Docstring description of the return value.")
  docstring: str | None = Field(None, description="Full docstring of the operation.")
  is_overload: bool = Field(False, description="Whether this is an overload of another function.")
  has_varargs: bool = Field(False, description="Whether the callable accepts variadic positional arguments.")
  is_public: bool = Field(True, description="Whether the symbol is part of the public API surface.")
  is_c_extension: bool = Field(False, description="Whether the symbol is implemented as a native C extension.")
  aliases: list[str] = Field(default_factory=list, description="Alternative qualified paths for the symbol.")

  @model_validator(mode="before")
  @classmethod
  def _normalize_operation_fields(
    cls,
    data: (
      dict[str, JsonValue | SnapshotArgument | list[SnapshotArgument]]
      | typing.Mapping[str, JsonValue | SnapshotArgument | list[SnapshotArgument]]
    ),
  ) -> dict[str, JsonValue | SnapshotArgument | list[SnapshotArgument]]:
    """Normalize v0.0.6 `params`, `returns_type`, `api_path`, and class `fields`.

    Args:
        data: Raw dictionary of operation properties.

    Returns:
        Normalized operation dictionary.
    """
    out: dict[str, JsonValue | SnapshotArgument | list[SnapshotArgument]] = dict(data)
    if "arguments" not in out and isinstance(out.get("params"), list):
      out["arguments"] = out["params"]
    if out.get("returns") is None and out.get("returns_type") is not None:
      out["returns"] = str(out["returns_type"])
    if not out.get("api_path") and isinstance(out.get("name"), str):
      out["api_path"] = out["name"]
    raw_kind = out.get("kind")
    if isinstance(raw_kind, str) and raw_kind.lower() == "class" and "fields" not in out:
      out["fields"] = out.get("arguments", [])
    raw_overloads = out.get("overloads")
    if isinstance(raw_overloads, list) and len(raw_overloads) > 0 and "is_overload" not in out:
      out["is_overload"] = True
    return out


class FrameworkSnapshot(BaseModel):
  """Represents a complete GhostRef snapshot of a machine learning framework."""

  model_config = ConfigDict(extra="ignore")

  framework_name: str = Field(..., description="Name of the framework (e.g., 'pytorch', 'jax').")
  version: str = Field(..., description="Version of the framework at the time of the snapshot.")
  schema_version: str = Field("2.0.0", description="Snapshot schema version.")
  generated_at: str | None = Field(None, description="ISO-8601 timestamp when the snapshot was extracted.")
  operations: list[SnapshotOperation] = Field(default_factory=list, description="List of operations in the framework.")
  categories: dict[str, list[SnapshotOperation]] = Field(
    default_factory=dict, description="Categorized operations map from v0.0.6 SnapshotEnvelope."
  )
  metadata: dict[str, JsonValue] = Field(default_factory=dict, description="Additional environment metadata.")

  @model_validator(mode="before")
  @classmethod
  def _normalize_envelope_fields(
    cls,
    data: (
      dict[str, JsonValue | SnapshotOperation | list[SnapshotOperation]]
      | typing.Mapping[str, JsonValue | SnapshotOperation | list[SnapshotOperation]]
    ),
  ) -> dict[str, JsonValue | SnapshotOperation | list[SnapshotOperation]]:
    """Normalize v0.0.6 `target` and `categories` envelope fields into `framework_name` and `operations`.

    Args:
        data: Raw dictionary of framework snapshot properties.

    Returns:
        Normalized framework snapshot dictionary.
    """
    out: dict[str, JsonValue | SnapshotOperation | list[SnapshotOperation]] = dict(data)
    if "framework_name" not in out and isinstance(out.get("target"), str):
      out["framework_name"] = out["target"]
    raw_cats = out.get("categories")
    if "operations" not in out and isinstance(raw_cats, dict):
      flat_ops: list[JsonValue] = []
      cleaned_cats: dict[str, JsonValue] = {}
      for cat_key, cat_items in raw_cats.items():
        if isinstance(cat_items, list):
          flat_ops.extend(cat_items)
          cleaned_cats[cat_key] = cat_items
      out["operations"] = flat_ops
      out["categories"] = cleaned_cats
    return out

  def module_hierarchy(self) -> dict[str, list[SnapshotOperation]]:
    """Group all snapshot operations by their parent module path.

    Returns:
        Dictionary mapping module dot-paths to the operations defined in that module.
    """
    hierarchy: dict[str, list[SnapshotOperation]] = {}
    for op in self.operations:
      full_path = op.api_path if op.api_path else op.name
      mod_path = full_path.rsplit(".", 1)[0] if "." in full_path else self.framework_name
      hierarchy.setdefault(mod_path, []).append(op)
    return hierarchy

  def classes(self) -> list[SnapshotOperation]:
    """Return all class definitions present in the snapshot.

    Returns:
        List of `SnapshotOperation` entries with `kind.lower() == 'class'`.
    """
    return [op for op in self.operations if op.kind.lower() == "class"]

  def functions(self) -> list[SnapshotOperation]:
    """Return all function and method definitions present in the snapshot.

    Returns:
        List of `SnapshotOperation` entries with `kind.lower() in ('function', 'method')`.
    """
    return [op for op in self.operations if op.kind.lower() in ("function", "method")]


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
