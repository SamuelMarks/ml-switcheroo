"""
Integration Test for Tracer Diffs.

Verifies that the engine produces trace events with populated
'before' and 'after' code snapshots.
"""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.core.tracer import TraceEventType, TraceLogger
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager


class MockSemantics(SemanticsManager):
  def __init__(self):
    self.data = {}
    self._reverse_index = {}
    self.import_data = {}
    self.framework_configs = {}
    self._validation_status = {}
    self._key_origins = {}

    # Inject "torch.abs" -> "jax.numpy.abs"
    self.data["abs"] = {"variants": {"torch": {"api": "torch.abs"}, "jax": {"api": "jax.numpy.abs"}}, "std_args": ["x"]}
    self._reverse_index["torch.abs"] = ("abs", self.data["abs"])


def test_conversion_trace_contains_diffs():
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=semantics, config=config)

  code = "y = torch.abs(x)"
  result = engine.run(code)

  assert result.success

  # Filter for AST Mutations
  mutations = [e for e in result.trace_events if e["type"] == TraceEventType.AST_MUTATION]

  # Debug: Print found descriptions if assertion fails
  found_descriptions = [m["description"] for m in mutations]

  # Locate the specific operation event
  # We look for "Operation (abs)" which is logged by CallMixin
  op_event = next((e for e in mutations if "Operation (abs)" in e["description"]), None)

  assert op_event is not None, f"Could not find 'Operation (abs)' in events: {found_descriptions}"

  assert op_event["metadata"]["before"].strip() == "torch.abs(x)"
  assert op_event["metadata"]["after"].strip() == "jax.numpy.abs(x)"


def test_lifecycle_strip_trace():
  semantics = MockSemantics()
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  engine = ASTEngine(semantics=semantics, config=config)

  # x.cpu() -> x (Stripping lifecycle)
  code = "y = x.cpu()"
  result = engine.run(code)

  mutations = [e for e in result.trace_events if e["type"] == TraceEventType.AST_MUTATION]

  # Should catch the strip event
  strip_event = next((e for e in mutations if "Lifecycle Strip" in e["description"]), None)

  assert strip_event is not None
  assert strip_event["metadata"]["before"].strip() == "x.cpu()"
  assert strip_event["metadata"]["after"].strip() == "x"
