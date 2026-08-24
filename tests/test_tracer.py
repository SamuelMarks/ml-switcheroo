"""Docstring."""

from ml_switcheroo.core.tracer import TraceLogger, TraceEventType, get_tracer, reset_tracer


def test_trace_logger_lifecycle():
  """Docstring."""
  logger = TraceLogger()
  phase_id = logger.start_phase("Test Phase", "Phase Description")
  assert phase_id is not None

  logger.log_match("src_func", "tgt_func", "Abs", lineno=10)
  logger.log_mutation("Call", "before()", "after()", lineno=11)
  logger.log_warning("Warning test", lineno=12)
  logger.log_inspection("node", "Skipped", "detail")
  logger.log_snapshot("Snapshot", "graph TD;")

  logger.end_phase()
  # End phase when stack is empty should be a no-op
  logger.end_phase()

  events = logger.export()

  # Phase start, Match, Mutation, Warning, Inspection, Snapshot, Phase End
  assert len(events) == 7

  start_evt = events[0]
  assert start_evt["type"] == TraceEventType.PHASE_START
  assert start_evt["description"] == "Test Phase"
  assert start_evt["parent_id"] is None

  match_evt = events[1]
  assert match_evt["type"] == TraceEventType.MATCH_SEMANTICS
  assert match_evt["metadata"]["abstract"] == "Abs"
  assert match_evt["parent_id"] == phase_id
  assert match_evt["lineno"] == 10

  mut_evt = events[2]
  assert mut_evt["type"] == TraceEventType.AST_MUTATION
  assert mut_evt["metadata"]["before"] == "before()"
  assert mut_evt["lineno"] == 11

  warn_evt = events[3]
  assert warn_evt["type"] == TraceEventType.ANALYSIS_WARNING
  assert warn_evt["metadata"]["level"] == "warning"

  insp_evt = events[4]
  assert insp_evt["type"] == TraceEventType.INSPECTION
  assert insp_evt["metadata"]["outcome"] == "Skipped"

  snap_evt = events[5]
  assert snap_evt["type"] == TraceEventType.AST_SNAPSHOT
  assert snap_evt["metadata"]["mermaid"] == "graph TD;"

  end_evt = events[6]
  assert end_evt["type"] == TraceEventType.PHASE_END
  assert end_evt["parent_id"] == phase_id


def test_global_tracer():
  """Docstring."""
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
