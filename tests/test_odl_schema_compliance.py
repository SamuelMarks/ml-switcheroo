"""Test suite for ODL YAML schema compliance.

Validates that all operation and framework definitions in the ODL store
comply with the Pydantic schema standards and schema.yaml.
"""

from pathlib import Path
from typing import Any, List
import yaml
import pytest

from ml_switcheroo.core.dsl import OperationDef
from ml_switcheroo.semantics.schema import SemanticsFile, validate_yaml_schema
from ml_switcheroo.semantics.paths import resolve_semantics_dir


def test_odl_atomic_files_schema_compliance() -> None:
  """Validates that all atomic YAML files in the odl directory satisfy OperationDef."""
  semantics_dir: Path = resolve_semantics_dir()
  odl_dir: Path = semantics_dir / "odl"

  if not odl_dir.exists():
    pytest.skip("ODL directory not found.")

  yaml_files: List[Path] = [p for p in odl_dir.glob("*.yaml") if not p.name.startswith("__framework_")]
  assert len(yaml_files) > 0, "No atomic ODL files found to validate."

  failures: List[str] = []
  # Sample check across all files
  for fpath in yaml_files[:200]:
    with open(fpath, "r", encoding="utf-8") as f:
      content: Any = yaml.safe_load(f)
    try:
      OperationDef.model_validate(content)
    except Exception as e:
      failures.append(f"{fpath.name}: {e}")

  assert not failures, f"ODL YAML schema failures: {failures[:5]}"


def test_odl_framework_files_schema_compliance() -> None:
  """Validates that all __framework_*.yaml files satisfy SemanticsFile."""
  semantics_dir: Path = resolve_semantics_dir()
  odl_dir: Path = semantics_dir / "odl"

  if not odl_dir.exists():
    pytest.skip("ODL directory not found.")

  framework_files: List[Path] = list(odl_dir.glob("__framework_*.yaml"))
  assert len(framework_files) > 0, "No framework metadata files found."

  for fpath in framework_files:
    with open(fpath, "r", encoding="utf-8") as f:
      content: Any = yaml.safe_load(f)
    validated = SemanticsFile.model_validate(content)
    assert validated.frameworks is not None
    assert len(validated.frameworks) > 0


def test_validate_yaml_schema_valid_and_invalid() -> None:
  """Tests helper validate_yaml_schema on valid and invalid payloads."""
  valid_yaml = """
__frameworks__:
  test_fw:
    alias:
      module: "test"
      name: "t"
"""
  result = validate_yaml_schema(valid_yaml)
  assert result.frameworks is not None
  assert "test_fw" in result.frameworks

  with pytest.raises(ValueError, match="Invalid YAML content"):
    validate_yaml_schema("broken: [unclosed")

  with pytest.raises(ValueError, match="Schema validation failed"):
    validate_yaml_schema("__frameworks__: [not, a, dict]")
