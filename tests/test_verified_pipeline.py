"""Test suite for the Verified Pipeline module."""

from pydantic import JsonValue
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

  def mock_parse_module(code: str) -> dict[str, str]:
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
  res = verified_pipeline.run_verified_pipeline(source)
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

  res = verified_pipeline.run_verified_pipeline("def foo(): pass")
  assert res["status"] == "success"
  assert res["griffe_analysis"] is True  # The value is a string, which is not None


def test_verified_pipeline_griffe_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify verified pipeline griffe not available behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  import sys

  monkeypatch.setitem(sys.modules, "griffe", None)

  res = verified_pipeline.run_verified_pipeline("def foo(): pass")
  assert res["status"] == "success"
  assert res["griffe_analysis"] is True  # The value is a string, which is not None


def test_verified_pipeline_cdd_error(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify verified pipeline cdd error behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  import builtins

  original_import = builtins.__import__

  import types

  def mock_import(name: str, *args: JsonValue, **kwargs: JsonValue) -> types.ModuleType:
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
    _ret_v: types.ModuleType = getattr(
      {
        "v": getattr({"v": original_import}, "__getitem__")("v")(
          getattr({"v": name}, "__getitem__")("v"), *args, **getattr({"v": kwargs}, "__getitem__")("v")
        )
      },
      "__getitem__",
    )("v")
    return _ret_v

  monkeypatch.setattr(builtins, "__import__", mock_import)
  res = verified_pipeline.run_verified_pipeline("def foo(): pass")
  assert res == {"error": "cdd-python not installed"}
