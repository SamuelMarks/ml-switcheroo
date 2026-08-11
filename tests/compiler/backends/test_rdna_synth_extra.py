"""Test suite for the Rdna Synth Extra module."""

import pytest
from ml_switcheroo.core.graph import LogicalGraph, LogicalNode
from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
from unittest.mock import MagicMock


def test_rdna_synth_raw_opcode():
  """Verifies the behavior of RDNA synth raw opcode."""
  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("rdna.v_add_f32", {})
  synth = RdnaSynthesizer(mock_semantics)
  g = LogicalGraph(nodes=[LogicalNode(id="n1", kind="rdna.v_add_f32")])
  synth.from_graph(g)


def test_rdna_synth_invalid_opcode():
  """Verifies the behavior of RDNA synth invalid opcode."""
  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("BadOp", {})
  mock_semantics.resolve_variant.return_value = {"api": "bad op code with spaces!"}
  synth = RdnaSynthesizer(mock_semantics)
  g = LogicalGraph(nodes=[LogicalNode(id="n1", kind="BadOp")])
  with pytest.raises(ValueError, match="Invalid RDNA opcode"):
    synth.from_graph(g)


def test_rdna_macros_coverage():
  """Test coverage for rdna macros."""
  from ml_switcheroo.core.compiler.backends.rdna.macros import (
    expand_dropout,
    expand_variable,
    expand_transpose,
    expand_conv_general_dilated,
    expand_adam,
    expand_l,
  )

  allocator = None
  assert len(expand_dropout(allocator, "n", {})) == 2
  assert len(expand_variable(allocator, "n", {})) == 2
  assert len(expand_transpose(allocator, "n", {})) == 2
  assert len(expand_conv_general_dilated(allocator, "n", {})) == 2
  assert len(expand_adam(allocator, "n", {})) == 2
  assert len(expand_l(allocator, "n", {})) == 2


def test_rdna_synth_suffix_macro():
  """Test coverage for rdna synthesizer suffix matching."""
  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("rdna.l", {})
  synth = RdnaSynthesizer(mock_semantics)
  g = LogicalGraph(nodes=[LogicalNode(id="n1", kind="rdna.l")])
  res = synth.from_graph(g)
  assert len(res) >= 1


def test_rdna_synth_macros_missing(monkeypatch):
  """Test rdna synth macros missing."""
  import os
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer

  monkeypatch.setattr(os.path, "exists", lambda x: False)
  synth = RdnaSynthesizer(None)
  assert synth.macro_registry == {}


def test_rdna_synth_macros_attr_missing(monkeypatch, tmp_path):
  """Test rdna synth macros attr missing."""
  import os
  import json
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer

  # create fake macros.json
  f = tmp_path / "macros.json"
  f.write_text(json.dumps({"test_op": "missing_macro_func"}))

  # patch os.path.dirname and os.path.exists
  def mock_dirname(p):
    """Mock dirname."""
    return str(tmp_path)

  def mock_exists(p):
    """Mock exists."""
    return p == str(f)

  monkeypatch.setattr(os.path, "dirname", mock_dirname)
  monkeypatch.setattr(os.path, "exists", mock_exists)

  synth = RdnaSynthesizer(None)
  assert synth.macro_registry == {}


def test_rdna_synth_output_no_sources():
  """Test rdna synth output no sources."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
  from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode

  synth = RdnaSynthesizer(None)
  graph = LogicalGraph("Test")
  graph.nodes.append(LogicalNode("out", "Output"))
  # input_map is empty, so sources is []
  nodes = synth.from_graph(graph)
  assert len(nodes) == 0


def test_rdna_synth_abstract_id_none():
  """Test rdna synth abstract id none."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
  from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode

  synth = RdnaSynthesizer(None)

  class FakeSem:
    """Fake sem."""

    def get_definition(self, kind):
      """Get definition."""
      return ("", {})  # abstract_id empty

  synth.semantics = FakeSem()
  graph = LogicalGraph("Test")
  graph.nodes.append(LogicalNode("n", "not_mapped"))
  nodes = synth.from_graph(graph)
  # 214->217 is hit because abstract_id == ""
  assert "Unmapped Op:" in nodes[0].text


def test_rdna_synth_to_cst_other_node():
  """Test rdna synth to cst other node."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment

  synth = RdnaSynthesizer(None)
  mod = synth.to_python([RdnaComment("test comment")])
  # 253->259: node is RdnaComment (not instruction, not label) -> stmt is None
  # 259->249: stmt is None so body_stmts.append is skipped
  assert len(mod.body) == 0


def test_rdna_synth_to_cst_label():
  """Test rdna synth to cst label."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaLabel

  synth = RdnaSynthesizer(None)
  mod = synth.to_python([RdnaLabel("lbl")])
  assert len(mod.body) == 1


def test_rdna_backend_semantics_provided():
  """Test rdna backend semantics provided."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaBackend
  from ml_switcheroo.semantics.manager import SemanticsManager

  sem = SemanticsManager()
  backend = RdnaBackend(semantics=sem)
  assert backend.synthesizer.semantics is sem
