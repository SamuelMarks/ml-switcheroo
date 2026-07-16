"""Test module."""

import pytest
from ml_switcheroo.core.graph import LogicalGraph, LogicalNode
from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
from unittest.mock import MagicMock


def test_rdna_synth_raw_opcode():
  """Test function."""
  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("rdna.v_add_f32", {})
  synth = RdnaSynthesizer(mock_semantics)
  g = LogicalGraph(nodes=[LogicalNode(id="n1", kind="rdna.v_add_f32")])
  synth.from_graph(g)


def test_rdna_synth_invalid_opcode():
  """Test function."""
  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("BadOp", {})
  mock_semantics.resolve_variant.return_value = {"api": "bad op code with spaces!"}
  synth = RdnaSynthesizer(mock_semantics)
  g = LogicalGraph(nodes=[LogicalNode(id="n1", kind="BadOp")])
  with pytest.raises(ValueError, match="Invalid RDNA opcode"):
    synth.from_graph(g)
