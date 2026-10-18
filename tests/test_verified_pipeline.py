"""Test suite for the Verified Pipeline module."""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from ml_switcheroo.ingestion import verified_pipeline


def test_verified_pipeline_dummy() -> None:
  """Verifies the behavior of verified pipeline dummy."""
  assert hasattr(verified_pipeline, "run_verified_pipeline")


def test_verified_pipeline_griffe_available(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify verified pipeline griffe available behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  import sys

  def mock_parse_module(code: str) -> Dict[str, str]:
    """Mock parse_module success.

    Args:
        code: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return {"parsed": code}

  class MockGriffe:
    """Mock Griffe module."""

    parse_module = mock_parse_module

  monkeypatch.setitem(sys.modules, "griffe", MockGriffe())

  source: str = "def foo(): pass"
  res: Dict[str, Any] = verified_pipeline.run_verified_pipeline(source)
  assert res["status"] == "success"
  assert res["ast_nodes"] == 1


@patch("ml_switcheroo.ingestion.verified_pipeline.ast.parse")
def test_verified_pipeline_ast_error(mock_ast_parse: MagicMock) -> None:
  """Verify verified pipeline ast error behavior.

  Args:
      mock_ast_parse: Test argument or fixture.
  """
  mock_ast_parse.side_effect = SyntaxError("test syntax error")
  with pytest.raises(SyntaxError):
    verified_pipeline.run_verified_pipeline("invalid code")


def test_verified_pipeline_griffe_error(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify verified pipeline griffe error behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """

  def mock_parse_module(code: str) -> None:
    """Mocks parse_module to throw an error.

    Args:
        code: Test argument or fixture.
    """
    raise ValueError("mock error")

  import sys

  class MockGriffe:
    """Mock Griffe module."""

    parse_module = mock_parse_module

  monkeypatch.setitem(sys.modules, "griffe", MockGriffe())

  res: Dict[str, Any] = verified_pipeline.run_verified_pipeline("def foo(): pass")
  assert res["status"] == "success"
  assert res["griffe_analysis"] is True  # The value is a string, which is not None


def test_verified_pipeline_griffe_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify verified pipeline griffe not available behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  import sys

  monkeypatch.setitem(sys.modules, "griffe", None)

  res: Dict[str, Any] = verified_pipeline.run_verified_pipeline("def foo(): pass")
  assert res["status"] == "success"
  assert res["griffe_analysis"] is True  # The value is a string, which is not None


def test_verified_pipeline_cdd_error(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify verified pipeline cdd error behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  import builtins

  original_import: Any = builtins.__import__

  def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """Mock import.

    Args:
        name: Test argument or fixture.
        args: Test argument or fixture.
        kwargs: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if name == "cdd":
      raise ImportError("Mocked ImportError")
    return original_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", mock_import)
  res: Dict[str, Any] = verified_pipeline.run_verified_pipeline("def foo(): pass")
  assert res == {"error": "cdd-python not installed"}
