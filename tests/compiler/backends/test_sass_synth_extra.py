"""Test module."""

import pytest
from ml_switcheroo.core.graph import LogicalGraph, LogicalNode
from ml_switcheroo.core.compiler.backends.sass.synthesizer import SassSynthesizer
from unittest.mock import MagicMock


def test_sass_synth_invalid_opcode() -> None:
  """Test function."""
  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("BadOp", {})
  mock_semantics.resolve_variant.return_value = {"api": "bad op code!"}
  synth = SassSynthesizer(mock_semantics)
  g = LogicalGraph(nodes=[LogicalNode(id="n1", kind="BadOp")])
  with pytest.raises(ValueError, match="Invalid SASS opcode"):
    synth.from_graph(g)


def test_sass_synth_liveness_tracking() -> None:
  """Test that liveness tracking frees registers."""
  from ml_switcheroo.core.compiler.ir import LogicalEdge

  mock_semantics = MagicMock()
  # Add opcode
  mock_semantics.get_definition.return_value = ("Add", {})
  mock_semantics.resolve_variant.return_value = {"api": "FADD"}
  synth = SassSynthesizer(mock_semantics)

  g = LogicalGraph()
  g.nodes = [LogicalNode(id="n1", kind="Input"), LogicalNode(id="n2", kind="Add")]
  g.edges = [LogicalEdge(source="n1", target="n2")]

  # Before from_graph, no liveness map
  assert synth.allocator._liveness_map == {}

  synth.from_graph(g)

  # After from_graph, n1 should be processed and its usage recorded (decremented to 0)
  assert synth.allocator._liveness_map["n1"] == 0

  # And the register allocated to n1 should be returned to the free pool
  # We can check if it's back in the free pool by checking the length of free pool
  assert len(synth.allocator._free_pool) == 254  # R0 and R1 were allocated, R0 freed => R0 is now at end of pool
