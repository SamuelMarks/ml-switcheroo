"""Test suite for the Trace module."""

from ml_switcheroo.core.tracer import TraceEventType, TraceLogger


def test_phase_nesting() -> None:
  """Verifies the behavior of phase nesting."""
  logger = TraceLogger()
  p1 = logger.start_phase("Parent")
  logger.start_phase("Child")
  logger.end_phase()
  logger.end_phase()
  events = logger.export()
  assert len(events) == 4
  assert events[0]["type"] == TraceEventType.PHASE_START
  assert events[1]["parent_id"] == p1
  assert events[2]["type"] == TraceEventType.PHASE_END


def test_trace_logging_integration() -> None:
  """Verifies the behavior of trace logging integration."""
  logger = TraceLogger()
  logger.log_match("torch.abs", "jax.numpy.abs", "abs")
  events = logger.export()
  assert len(events) == 1
  assert events[0]["type"] == TraceEventType.MATCH_SEMANTICS
  assert getattr(
    {
      "v": getattr(
        {"v": getattr({"v": getattr({"v": events}, "__getitem__")("v")[0]}, "__getitem__")("v")["metadata"]},
        "__getitem__",
      )("v")["source"]
    },
    "__getitem__",
  )("v") == getattr({"v": "torch.abs"}, "__getitem__")("v")


def test_snapshot_includes_source_code() -> None:
  """Verifies the behavior of snapshot includes source code."""
  logger = TraceLogger()
  logger.log_snapshot("Test Snap", "graph TD", "x = 1")
  events = logger.export()
  assert len(events) == 1
  assert events[0]["type"] == TraceEventType.AST_SNAPSHOT
  meta = events[0]["metadata"]
  assert getattr({"v": getattr({"v": meta}, "__getitem__")("v")["mermaid"]}, "__getitem__")("v") == getattr(
    {"v": "graph TD"}, "__getitem__"
  )("v")
  assert getattr({"v": getattr({"v": meta}, "__getitem__")("v")["code"]}, "__getitem__")("v") == getattr(
    {"v": "x = 1"}, "__getitem__"
  )("v")
