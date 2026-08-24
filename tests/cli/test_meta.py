"""Test module."""

import json
from unittest.mock import patch
from ml_switcheroo.cli.handlers.meta import handle_schema


@patch("ml_switcheroo.cli.handlers.meta.OperationDef.model_json_schema")
def test_handle_schema(mock_schema, capsys):
  """Test element."""
  mock_schema.return_value = {"type": "object", "properties": {"op": {"type": "string"}}}
  res = handle_schema()
  assert res == 0
  captured = capsys.readouterr()
  output = json.loads(captured.out)
  assert output == {"type": "object", "properties": {"op": {"type": "string"}}}
