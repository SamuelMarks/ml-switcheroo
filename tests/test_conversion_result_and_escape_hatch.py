"""Tests for ConversionResult and EscapeHatch components."""

from unittest.mock import MagicMock
import libcst as cst

from ml_switcheroo.core.conversion_result import ConversionResult
from ml_switcheroo.core.escape_hatch import EscapeHatch


def test_conversion_result_defaults_and_has_errors() -> None:
  """Test default field initialization and has_errors property on ConversionResult."""
  res_success = ConversionResult(code="y = 1")
  assert res_success.code == "y = 1"
  assert res_success.errors == []
  assert res_success.success is True
  assert res_success.trace_events == []
  assert res_success.has_errors is False

  res_failure = ConversionResult(code="", errors=["Failed to transpile node"], success=False)
  assert res_failure.has_errors is True


def test_escape_hatch_mark_failure_standard_statement() -> None:
  """Test mark_failure on standard CST statement node returning FlattenSentinel."""
  stmt = cst.parse_statement("y = unknown_func(x)")
  reason = "Unsupported dynamic operation"
  marked = EscapeHatch.mark_failure(stmt, reason)

  assert isinstance(marked, cst.FlattenSentinel)
  nodes = list(marked.nodes)
  assert len(nodes) == 2

  module = cst.Module(body=list(marked.nodes))
  code = module.code
  assert EscapeHatch.START_MARKER in code
  assert f"# Reason: {reason}" in code
  assert "y = unknown_func(x)" in code
  assert EscapeHatch.END_MARKER in code


def test_escape_hatch_mark_failure_fallback_on_exception() -> None:
  """Test mark_failure fallback when with_changes raises AttributeError or TypeError."""
  mock_node = MagicMock(spec=cst.CSTNode)
  mock_node.with_changes.side_effect = TypeError("Cannot change leading lines")

  res = EscapeHatch.mark_failure(mock_node, "Some failure reason")
  assert res is mock_node
