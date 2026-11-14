"""Unit tests for `handle_schema` (`ml_switcheroo.cli.handlers.meta`)."""

import json

import pytest

from ml_switcheroo.cli.handlers.meta import handle_schema


def test_handle_schema_outputs_operation_def_json_schema(capsys: pytest.CaptureFixture[str]) -> None:
  """Verify `handle_schema` emits valid JSON schema for `OperationDef`.

  Args:
      capsys: Pytest stdout/stderr capture fixture.
  """
  assert handle_schema() == 0
  schema = json.loads(capsys.readouterr().out)
  assert schema["title"] == "OperationDef"
  assert "operation" in schema["properties"]
