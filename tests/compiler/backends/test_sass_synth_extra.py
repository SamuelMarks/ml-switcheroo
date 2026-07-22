"""Test suite for the Sass Synth Extra module."""

import pytest
from ml_switcheroo.core.graph import LogicalGraph, LogicalNode
from ml_switcheroo.core.compiler.backends.sass.synthesizer import SassSynthesizer
from unittest.mock import MagicMock


def test_sass_synth_invalid_opcode() -> None:
  """Verifies the behavior of SASS synth invalid opcode."""
  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("BadOp", {})
  mock_semantics.resolve_variant.return_value = {"api": "bad op code!"}
  synth = SassSynthesizer(mock_semantics)
  g = LogicalGraph(nodes=[LogicalNode(id="n1", kind="BadOp")])
  with pytest.raises(ValueError, match="Invalid SASS opcode"):
    synth.from_graph(g)


def test_sass_synth_liveness_tracking() -> None:
  """Verifies the behavior of SASS synth liveness tracking."""
  from ml_switcheroo.core.compiler.ir import LogicalEdge

  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("Add", {})
  mock_semantics.resolve_variant.return_value = {"api": "FADD"}
  synth = SassSynthesizer(mock_semantics)
  g = LogicalGraph()
  g.nodes = [LogicalNode(id="n1", kind="Input"), LogicalNode(id="n2", kind="Add")]
  g.edges = [LogicalEdge(source="n1", target="n2")]
  assert synth.allocator._liveness_map == {}
  synth.from_graph(g)
  assert synth.allocator._liveness_map["n1"] == 0
  assert len(synth.allocator._free_pool) == 254
