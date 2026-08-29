"""Test suite for the Trace module."""

import typing

from ml_switcheroo.core.tracer import TraceEventType, TraceLogger


def test_phase_nesting() -> None:
  """Verifies the behavior of phase nesting."""
  logger = TraceLogger()
  p1 = logger.start_phase("Parent")
  logger.start_phase("Child")
  logger.end_phase()
  logger.end_phase()
  events: list[dict[str, typing.Any]] = logger.export()
  assert len(events) == 4
  assert events[0]["type"] == TraceEventType.PHASE_START
  assert events[1]["parent_id"] == p1
  assert events[2]["type"] == TraceEventType.PHASE_END


def test_trace_logging_integration() -> None:
  """Verifies the behavior of trace logging integration."""
  logger = TraceLogger()
  logger.log_match("torch.abs", "jax.numpy.abs", "abs")
  events: list[dict[str, typing.Any]] = logger.export()
  assert len(events) == 1
  assert events[0]["type"] == TraceEventType.MATCH_SEMANTICS
  assert events[0]["metadata"]["source"] == "torch.abs"


def test_snapshot_includes_source_code() -> None:
  """Verifies the behavior of snapshot includes source code."""
  logger = TraceLogger()
  logger.log_snapshot("Test Snap", "graph TD", "x = 1")
  events: list[dict[str, typing.Any]] = logger.export()
  assert len(events) == 1
  assert events[0]["type"] == TraceEventType.AST_SNAPSHOT
  meta: dict[str, typing.Any] = events[0]["metadata"]
  assert meta["mermaid"] == "graph TD"
  assert meta["code"] == "x = 1"
