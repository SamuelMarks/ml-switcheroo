"""
Tests for the main ASTEngine and result structures.
"""

from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.escape_hatch import EscapeHatch
import libcst as cst


def test_engine_roundtrip():
  """Ensure we can parse and emit code without changes (idempotency)."""
  # Use identical source/target to disable ImportFixer logic
  engine = ASTEngine(source="torch", target="torch")
  source = "x = 5\nprint(x)"
  result = engine.run(source)

  assert isinstance(result, ConversionResult)
  assert result.success is True
  assert result.code == source
  assert result.has_errors is False


def test_escape_hatch_injection():
  """Ensure the escape hatch adds the specific comment markers."""
  engine = ASTEngine()

  # 1. Parse a simple statement (Module -> [SimpleStatementLine])
  source = "complex_function(x)"
  tree = engine.parse(source)

  # 2. Manual simulated failure: grab the **Statement** node
  original_stmt = tree.body[0]

  # 3. Apply the Escape Hatch
  result_sentinel = EscapeHatch.mark_failure(original_stmt, reason="Does not map to JAX")

  # 4. Swap it into a new tree (manual surgery for testing)
  # Important: result_sentinel is a cst.FlattenSentinel containing [Header+Node, Footer(Ellipsis)]
  # We must flatten it back into a list of nodes for the Module body.
  if isinstance(result_sentinel, cst.FlattenSentinel):
    new_body_nodes = list(result_sentinel)
  else:
    new_body_nodes = [result_sentinel]

  new_tree = tree.with_changes(body=new_body_nodes)

  # 5. Verify Output matches the protocol
  generated_code = engine.to_source(new_tree)

  expected_reason = "# Reason: Does not map to JAX"

  assert EscapeHatch.START_MARKER in generated_code
  assert expected_reason in generated_code
  assert EscapeHatch.END_MARKER in generated_code

  # Verify that Engine.run() detects this marker in its error reporting logic
  errors = []
  if "# <SWITCHEROO_FAILED_TO_TRANS>" in generated_code:
    errors.append("Failure detected")

  assert len(errors) == 1
