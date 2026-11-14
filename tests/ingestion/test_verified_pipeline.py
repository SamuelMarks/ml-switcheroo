"""Test suite for the Verified Pipeline module."""

from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo.ingestion import verified_pipeline


def test_verified_pipeline_dummy() -> None:
  """Verify verified pipeline dummy produces the expected state."""
  assert callable(verified_pipeline.run_verified_pipeline)
  res = verified_pipeline.run_verified_pipeline("x = 1\ny = 2\n")
  assert res["status"] == "success"
  assert res["ast_nodes"] == 2
  assert res["structural_normalized"] is True
  assert res["griffe_analysis"] is True


def test_verified_pipeline_griffe_available(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `monkeypatch.setitem` handles verified pipeline griffe available as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  import sys

  def mock_parse_module(code: str) -> dict[str, str]:
    """Mock parse_module success.

    Args:
        code: Python or DSL source code string to parse or transpile.

    Returns:
        Resulting `dict[str, str]` from mock parse module.
    """
    return {"parsed": code}

  class MockGriffe:
    """Mock Griffe module."""

    parse_module = mock_parse_module

  monkeypatch.setitem(sys.modules, "griffe", MockGriffe())

  source: str = "def foo(): pass"
  res = verified_pipeline.run_verified_pipeline(source)
  assert res["status"] == "success"
  assert res["ast_nodes"] == 1


@patch("ml_switcheroo.ingestion.verified_pipeline.ast.parse")
def test_verified_pipeline_ast_error(mock_ast_parse: MagicMock) -> None:
  """Verify `SyntaxError` raises `SyntaxError` during verified pipeline AST error.

  Args:
      mock_ast_parse: Input `mock_ast_parse` parameter value (`MagicMock`).
  """
  mock_ast_parse.side_effect = SyntaxError("test syntax error")
  with pytest.raises(SyntaxError):
    verified_pipeline.run_verified_pipeline("invalid code")


def test_verified_pipeline_griffe_error(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `monkeypatch.setitem` handles verified pipeline griffe error as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """

  def mock_parse_module(code: str) -> None:
    """Mocks parse_module to throw an error.

    Args:
        code: Python or DSL source code string to parse or transpile.
    """
    raise ValueError("mock error")

  import sys

  class MockGriffe:
    """Mock Griffe module."""

    parse_module = mock_parse_module

  monkeypatch.setitem(sys.modules, "griffe", MockGriffe())

  res = verified_pipeline.run_verified_pipeline("def foo(): pass")
  assert res["status"] == "success"
  assert res["griffe_analysis"] is True  # The value is a string, which is not None


def test_verified_pipeline_griffe_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `monkeypatch.setitem` handles verified pipeline griffe not available as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  import sys

  monkeypatch.setitem(sys.modules, "griffe", None)

  res = verified_pipeline.run_verified_pipeline("def foo(): pass")
  assert res["status"] == "success"
  assert res["griffe_analysis"] is True  # The value is a string, which is not None


def test_verified_pipeline_cdd_error(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `verified_pipeline.run_verified_pipeline` handles verified pipeline cdd error as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  import sys

  monkeypatch.setitem(sys.modules, "cdd", None)
  res = verified_pipeline.run_verified_pipeline("def foo(): pass")
  assert res == {"error": "cdd-python not installed"}
