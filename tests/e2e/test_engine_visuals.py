"""Test suite for the Engine Visuals module."""

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.core.tracer import TraceEventType


def test_engine_emits_valid_mermaid() -> None:
  """Verify `RuntimeConfig` handles engine emits valid mermaid as expected."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(config=config)
  code = "x = 1"
  result = engine.run(code)
  assert result.success
  snapshots = [e for e in result.trace_events if e["type"] == TraceEventType.AST_SNAPSHOT]
  assert len(snapshots) > 0
  snap = snapshots[0]
  metadata = snap.get("metadata")
  assert isinstance(metadata, dict)
  mermaid_code = metadata.get("mermaid")
  assert isinstance(mermaid_code, str)
  assert mermaid_code.startswith("graph TD")
  assert "classDef" in mermaid_code
  assert "::modNode" in mermaid_code
