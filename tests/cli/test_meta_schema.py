"""
Tests for ODL Schema Export.

Verifies that:
1. The `schema` command outputs valid JSON.
2. The JSON represents the `OperationDef` Pydantic model.
3. Key fields (std_args, variants) are present in the schema.
"""

import json
import pytest
from ml_switcheroo.cli.__main__ import main
from ml_switcheroo.core.dsl import OperationDef


def test_schema_command_integration(capsys):
  """
  Scenario: User runs `ml_switcheroo schema`.
  Expectation: JSON Schema printed to stdout.
  """
  # 1. Run via main CLI entry point
  # We pass explicit arguments list to simulate argv
  args = ["schema"]
  ret_code = main(args)

  assert ret_code == 0

  # 2. Capture and Validate Output
  output = capsys.readouterr().out
  assert output.strip(), "Output should not be empty"

  try:
    schema = json.loads(output)
  except json.JSONDecodeError as e:
    pytest.fail(f"Output is not valid JSON: {e}")

  # 3. Validate Schema Content matches Pydantic Source of Truth
  # Ensure it matches the structure of OperationDef
  expected_ref_keys = ["description", "properties", "title"]
  for k in expected_ref_keys:
    assert k in schema

  assert schema["title"] == "OperationDef"

  props = schema.get("properties", {})
  assert "operation" in props
  assert "std_args" in props
  assert "variants" in props
  assert "scaffold_plugins" in props

  # Check deep structure (e.g. variants should be a Dict)
  # Note: Pydantic v2 schemas use $defs, but properties structure remains
  assert schema.get("type") == "object"
