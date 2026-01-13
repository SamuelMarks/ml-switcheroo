"""
Integration Tests for Engine Visuals.

Verifies that the ASTEngine populates the trace event with a valid
Mermaid graph string using the MermaidGenerator.
"""

import pytest
from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.tracer import TraceEventType


def test_engine_emits_valid_mermaid():
  """
  Scenario: Run simple conversion.
  Expectation: Trace events of type AST_SNAPSHOT contain 'mermaid' field starting with 'graph TD'.
  """
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  # No semantics needed for structural parsing test
  engine = ASTEngine(config=config)

  code = "x = 1"
  result = engine.run(code)

  assert result.success

  # Find snapshot event
  snapshots = [e for e in result.trace_events if e["type"] == TraceEventType.AST_SNAPSHOT]
  assert len(snapshots) > 0

  # Verify content
  snap = snapshots[0]
  mermaid_code = snap["metadata"]["mermaid"]

  assert mermaid_code is not None
  assert mermaid_code.startswith("graph TD")
  assert "classDef" in mermaid_code
  # check that our specific generator logic ran (e.g. ::modNode style)
  assert "::modNode" in mermaid_code
