"""Unit tests for `ConversionResult` and `EscapeHatch` (`ml_switcheroo.core`)."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.conversion_result import ConversionResult
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.core.escape_hatch import EscapeHatch

_NL = chr(10)


def test_conversion_result_defaults_and_flags() -> None:
  """Verify default field initialization and `has_errors` / `has_warnings` properties on `ConversionResult`."""
  res_success = ConversionResult(code="y = 1")
  assert res_success.code == "y = 1"
  assert res_success.errors == []
  assert res_success.warnings == []
  assert res_success.success is True
  assert res_success.trace_events == []
  assert res_success.has_errors is False
  assert res_success.has_warnings is False

  res_failure = ConversionResult(code="", errors=["Failed to transpile node"], success=False)
  assert res_failure.has_errors is True

  res_warning = ConversionResult(code="y = 1", warnings=["Dynamic construct detected"])
  assert res_warning.has_warnings is True


def test_escape_hatch_mark_failure_statement_and_expression_fallbacks() -> None:
  """Verify `EscapeHatch.mark_failure` wraps statements with sentinel comments and returns expressions intact."""
  stmt = cst.parse_statement("y = unknown_func(x)")
  assert isinstance(stmt, cst.SimpleStatementLine)
  reason = "Unsupported dynamic operation"
  marked = EscapeHatch.mark_failure(stmt, reason)

  assert isinstance(marked, cst.FlattenSentinel)
  nodes: list[cst.SimpleStatementLine] = [cst.ensure_type(n, cst.SimpleStatementLine) for n in marked.nodes]
  assert len(nodes) == 2

  code = cst.Module(body=nodes).code
  assert EscapeHatch.START_MARKER in code
  assert f"# Reason: {reason}" in code
  assert "y = unknown_func(x)" in code
  assert EscapeHatch.END_MARKER in code

  # Non-statement CST expressions lack `leading_lines` and return unchanged
  expr_node = cst.Name("x")
  assert EscapeHatch.mark_failure(expr_node, "expression fallback") is expr_node

  # Statement mock raising TypeError on `with_changes` falls back to returning the node unchanged
  mock_node = MagicMock(spec=cst.SimpleStatementLine)
  mock_node.leading_lines = []
  mock_node.with_changes.side_effect = TypeError("Cannot change leading lines")
  assert EscapeHatch.mark_failure(mock_node, "Some failure reason") is mock_node


def test_conversion_result_collects_static_safety_warnings() -> None:
  """Verify `ASTEngine.run` populates static safety warnings in `ConversionResult`."""
  engine = ASTEngine.from_options(source="torch", target="jax")
  code_with_dynamic = _NL.join(
    [
      "import torch",
      "def func(x):",
      "    if x.item() > 0:",
      "        return x[x > 0]",
      "    return x",
      "",
    ]
  )
  res = engine.run(code_with_dynamic)
  assert res.has_warnings is True
  assert any("DYNAMIC_SHAPE" in w for w in res.warnings)
  assert any("VALUE_DEPENDENT_CONTROL_FLOW" in w for w in res.warnings)
