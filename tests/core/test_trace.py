"""
Tests for the Tracing System.
"""

import pytest
from ml_switcheroo.core.tracer import TraceLogger, TraceEventType


def test_phase_nesting():
  logger = TraceLogger()

  p1 = logger.start_phase("Parent")
  p2 = logger.start_phase("Child")
  logger.end_phase()  # End Child
  logger.end_phase()  # End Parent

  events = logger.export()

  # 4 events: Start P, Start C, End C, End P
  assert len(events) == 4
  assert events[0]["type"] == TraceEventType.PHASE_START
  assert events[1]["parent_id"] == p1
  assert events[2]["type"] == TraceEventType.PHASE_END


def test_trace_logging_integration():
  """Verify log_match records correct metadata."""
  logger = TraceLogger()
  logger.log_match("torch.abs", "jax.numpy.abs", "abs")

  events = logger.export()
  assert len(events) == 1
  assert events[0]["type"] == TraceEventType.MATCH_SEMANTICS
  assert events[0]["metadata"]["source"] == "torch.abs"
