"""Unit tests for `TraceLogger` lifecycle and global tracer helpers (`ml_switcheroo.core.tracer`)."""

from ml_switcheroo.core.tracer import TraceEventType, TraceLogger, get_tracer, reset_tracer


def test_trace_logger_lifecycle() -> None:
  """Verify `TraceLogger` records structured phase, match, mutation, warning, inspection, and snapshot events."""
  logger = TraceLogger()
  # Ending a phase on an empty stack is a safe no-op
  logger.end_phase()

  phase_id = logger.start_phase("Test Phase", "Phase Description")
  assert phase_id is not None

  logger.log_match("src_func", "tgt_func", "Abs", lineno=10)
  logger.log_mutation("Call", "before()", "after()", lineno=11)
  logger.log_warning("Warning test", lineno=12)
  logger.log_inspection("node", "Skipped", "detail")
  logger.log_snapshot("Snapshot", "graph TD;")

  logger.end_phase()
  logger.end_phase()

  events = logger.export()
  assert len(events) == 7

  start_evt = events[0]
  assert start_evt["type"] == TraceEventType.PHASE_START
  assert start_evt["description"] == "Test Phase"
  assert start_evt["parent_id"] is None

  match_evt = events[1]
  assert match_evt["type"] == TraceEventType.MATCH_SEMANTICS
  match_meta = match_evt["metadata"]
  assert isinstance(match_meta, dict)
  assert match_meta["abstract"] == "Abs"
  assert match_evt["parent_id"] == phase_id
  assert match_evt["lineno"] == 10

  mut_evt = events[2]
  assert mut_evt["type"] == TraceEventType.AST_MUTATION
  mut_meta = mut_evt["metadata"]
  assert isinstance(mut_meta, dict)
  assert mut_meta["before"] == "before()"
  assert mut_evt["lineno"] == 11

  warn_evt = events[3]
  assert warn_evt["type"] == TraceEventType.ANALYSIS_WARNING
  warn_meta = warn_evt["metadata"]
  assert isinstance(warn_meta, dict)
  assert warn_meta["level"] == "warning"

  insp_evt = events[4]
  assert insp_evt["type"] == TraceEventType.INSPECTION
  insp_meta = insp_evt["metadata"]
  assert isinstance(insp_meta, dict)
  assert insp_meta["outcome"] == "Skipped"

  snap_evt = events[5]
  assert snap_evt["type"] == TraceEventType.AST_SNAPSHOT
  snap_meta = snap_evt["metadata"]
  assert isinstance(snap_meta, dict)
  assert snap_meta["mermaid"] == "graph TD;"

  end_evt = events[6]
  assert end_evt["type"] == TraceEventType.PHASE_END
  assert end_evt["parent_id"] == phase_id


def test_global_tracer() -> None:
  """Verify `get_tracer` returns a singleton instance reset by `reset_tracer`."""
  reset_tracer()
  t1 = get_tracer()
  t2 = get_tracer()
  assert t1 is t2

  t1.start_phase("Test")
  assert len(t1.export()) == 1

  reset_tracer()
  t3 = get_tracer()
  assert t3 is not t1
  assert len(t3.export()) == 0
