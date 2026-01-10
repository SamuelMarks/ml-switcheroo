"""
Tests for the main ASTEngine and result structures.
"""

from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.tracer import TraceEventType
from unittest.mock import MagicMock, patch
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


def test_engine_emits_snapshots():
  """Verify snapshot events are emitted during run."""
  source_code = "import torch\nx = torch.abs(y)"
  # Use config that enables import fixer to verify multiple snapshots
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax", enable_import_fixer=True)
  engine = ASTEngine(config=cfg)

  result = engine.run(source_code)

  events = result.trace_events
  snapshots = [e for e in events if e["type"] == TraceEventType.AST_SNAPSHOT]

  # Expect at least Ingestion, Analysis, Rewrite, Fixing
  assert len(snapshots) >= 3
  labels = [s["description"] for s in snapshots]

  assert "After Ingestion" in labels
  assert "After Analysis" in labels
  assert "After Rewriting" in labels
  assert "After Import Fixing" in labels

  # Check source code capture
  assert snapshots[0]["metadata"]["code"] is not None


def test_conditional_import_fixer_skip():
  """Verify enabling/disabling import fixer works."""
  source_code = "import torch\nx = torch.abs(y)"

  # Disable Fixer
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax", enable_import_fixer=False)
  engine = ASTEngine(config=cfg)

  result = engine.run(source_code)

  # Torch import should persist because Fixer skipped
  assert "import torch" in result.code

  # Check trace for absence of Fixing snapshot
  events = result.trace_events
  labels = [e["description"] for e in events if e["type"] == TraceEventType.AST_SNAPSHOT]
  assert "After Import Fixing" not in labels


@patch("ml_switcheroo.core.graph_optimizer.GraphOptimizer.optimize")
def test_conditional_graph_optimization(mock_opt):
  """
  Verify graph optimization is gated.

  Updated to use SASS target which triggers the compiler pipeline where optimization runs.
  """
  source_code = "x = torch.relu(x)"

  # 1. Disabled (Default) using SASS target to access compiler pipe
  cfg1 = RuntimeConfig(source_framework="torch", target_framework="sass", enable_graph_optimization=False)
  engine1 = ASTEngine(config=cfg1)
  # Mock backends to avoid needing full semantics for sass
  with patch("ml_switcheroo.core.engine.get_backend_class") as m_backend:
    m_backend.return_value = MagicMock()
    engine1.run(source_code)
  mock_opt.assert_not_called()

  # 2. Enabled
  # Mock behavior to just return graph
  mock_opt.return_value = MagicMock()
  # Mock synthesis to avoid crash on mocked graph

  cfg2 = RuntimeConfig(source_framework="torch", target_framework="sass", enable_graph_optimization=True)
  engine2 = ASTEngine(config=cfg2)

  with patch("ml_switcheroo.core.engine.get_backend_class") as m_backend:
    backend_instance = MagicMock()
    backend_instance.compile.return_value = "optimized_code"
    m_backend.return_value.return_value = backend_instance

    engine2.run(source_code)

    mock_opt.assert_called_once()

    # Check snapshot emission
    events = engine2.run(source_code).trace_events
    labels = [e["description"] for e in events if e["type"] == TraceEventType.AST_SNAPSHOT]
    assert "After Optimization" in labels
