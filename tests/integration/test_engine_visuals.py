"""Test suite for the Engine Visuals module."""

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.tracer import TraceEventType


def test_engine_emits_valid_mermaid():
  """Verifies the behavior of engine emits valid mermaid."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(config=config)
  code = "x = 1"
  result = engine.run(code)
  assert result.success
  snapshots = [e for e in result.trace_events if e["type"] == TraceEventType.AST_SNAPSHOT]
  assert len(snapshots) > 0
  snap = snapshots[0]
  mermaid_code = snap["metadata"]["mermaid"]
  assert mermaid_code is not None
  assert mermaid_code.startswith("graph TD")
  assert "classDef" in mermaid_code
  assert "::modNode" in mermaid_code
