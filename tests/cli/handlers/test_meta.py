"""Test suite for the Meta module."""

from ml_switcheroo.cli.handlers.meta import handle_schema
import json


def test_handle_schema(capsys):
  """Handles schema."""
  assert handle_schema() == 0
  captured = capsys.readouterr()
  data = json.loads(captured.out)
  assert isinstance(data, dict)
  assert "title" in data
