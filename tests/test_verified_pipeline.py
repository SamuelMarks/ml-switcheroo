"""Test suite for the Verified Pipeline module."""

from ml_switcheroo.ingestion import verified_pipeline
from unittest.mock import patch
import pytest


def test_verified_pipeline_dummy():
  """Verifies the behavior of verified pipeline dummy."""
  assert hasattr(verified_pipeline, "run_verified_pipeline")


def test_verified_pipeline_griffe_available():
  """Test pipeline when Griffe is available and parses successfully."""
  source = "def foo(): pass"
  res = verified_pipeline.run_verified_pipeline(source)
  assert res["status"] == "success"
  assert res["ast_nodes"] == 1


@patch("ml_switcheroo.ingestion.verified_pipeline.ast.parse")
def test_verified_pipeline_ast_error(mock_ast_parse):
  """Test pipeline when AST parsing fails."""
  mock_ast_parse.side_effect = SyntaxError("test syntax error")
  with pytest.raises(SyntaxError):
    verified_pipeline.run_verified_pipeline("invalid code")


def test_verified_pipeline_griffe_error(monkeypatch):
  """Test pipeline when Griffe throws an error during parsing."""

  def mock_parse_module(code):
    """Mocks parse_module to throw an error."""
    raise ValueError("mock error")

  import sys

  class MockGriffe:
    """Mock Griffe module."""

    parse_module = mock_parse_module

  monkeypatch.setitem(sys.modules, "griffe", MockGriffe())

  res = verified_pipeline.run_verified_pipeline("def foo(): pass")
  assert res["status"] == "success"
  assert res["griffe_analysis"] is True  # The value is a string, which is not None


def test_verified_pipeline_griffe_not_available(monkeypatch):
  """Test pipeline when Griffe is missing."""
  import sys

  monkeypatch.setitem(sys.modules, "griffe", None)

  res = verified_pipeline.run_verified_pipeline("def foo(): pass")
  assert res["status"] == "success"
  assert res["griffe_analysis"] is True  # The value is a string, which is not None


def test_verified_pipeline_cdd_error(monkeypatch):
  """Test pipeline when cdd is not installed."""
  import builtins

  original_import = builtins.__import__

  def mock_import(name, *args, **kwargs):
    """Mock import."""
    if name == "cdd":
      raise ImportError("Mocked ImportError")
    return original_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", mock_import)
  res = verified_pipeline.run_verified_pipeline("def foo(): pass")
  assert res == {"error": "cdd-python not installed"}
