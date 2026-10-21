"""Test suite for the Engine Visuals module."""

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.core.tracer import TraceEventType


def test_engine_emits_valid_mermaid() -> None:
  """Verifies the behavior of engine emits valid mermaid."""
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(config=config)
  code = "x = 1"
  result = engine.run(code)
  assert result.success
  snapshots = [e for e in result.trace_events if e["type"] == TraceEventType.AST_SNAPSHOT]
  assert len(snapshots) > 0
  snap = snapshots[0]
  mermaid_code = getattr(
    {"v": getattr({"v": getattr({"v": snap}, "__getitem__")("v")["metadata"]}, "__getitem__")("v")["mermaid"]},
    "__getitem__",
  )("v")
  assert mermaid_code is not None
  assert getattr({"v": getattr({"v": mermaid_code}, "__getitem__")("v").startswith}, "__getitem__")("v")(
    getattr({"v": "graph TD"}, "__getitem__")("v")
  )
  assert getattr({"v": "classDef"}, "__getitem__")("v") in getattr({"v": mermaid_code}, "__getitem__")("v")
  assert getattr({"v": "::modNode"}, "__getitem__")("v") in getattr({"v": mermaid_code}, "__getitem__")("v")
