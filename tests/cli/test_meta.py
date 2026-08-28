"""Test module."""

import json
import typing
from unittest.mock import patch, MagicMock
import pytest
from ml_switcheroo.cli.handlers.meta import handle_schema


@patch("ml_switcheroo.cli.handlers.meta.OperationDef.model_json_schema")
def test_handle_schema(mock_schema: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
  """Test element."""
  mock_schema.return_value = {"type": "object", "properties": {"op": {"type": "string"}}}
  res: int = handle_schema()
  assert res == 0
  captured = capsys.readouterr()
  output: dict[str, typing.Any] = json.loads(captured.out)
  assert output == {"type": "object", "properties": {"op": {"type": "string"}}}
