"""Test suite for the Engine Visuals module."""

from tests.conftest import _test_getattr

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
  mermaid_code = _test_getattr(
    {
      "v": _test_getattr({"v": _test_getattr({"v": snap}, "__getitem__")("v")["metadata"]}, "__getitem__")("v")["mermaid"]
    },
    "__getitem__",
  )("v")
  assert mermaid_code is not None
  assert _test_getattr({"v": _test_getattr({"v": mermaid_code}, "__getitem__")("v").startswith}, "__getitem__")("v")(
    _test_getattr({"v": "graph TD"}, "__getitem__")("v")
  )
  assert _test_getattr({"v": "classDef"}, "__getitem__")("v") in _test_getattr({"v": mermaid_code}, "__getitem__")("v")
  assert _test_getattr({"v": "::modNode"}, "__getitem__")("v") in _test_getattr({"v": mermaid_code}, "__getitem__")("v")
