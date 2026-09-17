"""Test suite for the Rdna Synth Extra module."""

from unittest.mock import MagicMock

import pytest

from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
from ml_switcheroo.core.graph import LogicalGraph, LogicalNode


def test_rdna_synth_raw_opcode():
  """Verifies the behavior of RDNA synth raw opcode."""
  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("rdna.v_add_f32", {})
  synth = RdnaSynthesizer(mock_semantics)
  g = LogicalGraph(nodes={n.id: n for n in [LogicalNode(id="n1", op_type="rdna.v_add_f32")]})
  synth.from_graph(g)


def test_rdna_synth_invalid_opcode():
  """Verifies the behavior of RDNA synth invalid opcode."""
  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("BadOp", {})
  mock_semantics.resolve_variant.return_value = {"api": "bad op code with spaces!"}
  synth = RdnaSynthesizer(mock_semantics)
  g = LogicalGraph(nodes={n.id: n for n in [LogicalNode(id="n1", op_type="BadOp")]})
  with pytest.raises(ValueError, match="Invalid RDNA opcode"):
    synth.from_graph(g)


def test_rdna_macros_coverage():
  """Docstring."""
  from ml_switcheroo.core.compiler.backends.rdna.macros import (
    expand_adam,
    expand_conv_general_dilated,
    expand_dropout,
    expand_l,
    expand_transpose,
    expand_variable,
  )

  allocator = None
  assert len(expand_dropout(allocator, "n", {})) == 2
  assert len(expand_variable(allocator, "n", {})) == 2
  assert len(expand_transpose(allocator, "n", {})) == 2
  assert len(expand_conv_general_dilated(allocator, "n", {})) == 2
  assert len(expand_adam(allocator, "n", {})) == 2
  assert len(expand_l(allocator, "n", {})) == 2


def test_rdna_synth_suffix_macro():
  """Docstring."""
  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("rdna.l", {})
  synth = RdnaSynthesizer(mock_semantics)
  g = LogicalGraph(nodes={n.id: n for n in [LogicalNode(id="n1", op_type="rdna.l")]})
  res = synth.from_graph(g)
  assert len(res) >= 1


def test_rdna_synth_macros_missing(monkeypatch):
  """Docstring."""
  import os

  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer

  monkeypatch.setattr(os.path, "exists", lambda x: False)
  synth = RdnaSynthesizer(None)
  assert synth.macro_registry == {}


def test_rdna_synth_macros_attr_missing(monkeypatch, tmp_path):
  """Docstring."""
  import json
  import os

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
  """Docstring."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
  from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode

  synth = RdnaSynthesizer(None)
  graph = LogicalGraph("Test")
  graph.add_node(LogicalNode("out", "Output"))
  # input_map is empty, so sources is []
  nodes = synth.from_graph(graph)
  assert len(nodes) == 0


def test_rdna_synth_abstract_id_none():
  """Docstring."""
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
  graph.add_node(LogicalNode("n", "not_mapped"))
  nodes = synth.from_graph(graph)
  # 214->217 is hit because abstract_id == ""
  assert "Unmapped Op:" in nodes[0].text


def test_rdna_synth_to_cst_other_node():
  """Docstring."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment

  synth = RdnaSynthesizer(None)
  mod = synth.to_python([RdnaComment("test comment")])
  # 253->259: node is RdnaComment (not instruction, not label) -> stmt is None
  # 259->249: stmt is None so body_stmts.append is skipped
  assert len(mod.body) == 0


def test_rdna_synth_to_cst_label():
  """Docstring."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaSynthesizer
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaLabel

  synth = RdnaSynthesizer(None)
  mod = synth.to_python([RdnaLabel("lbl")])
  assert len(mod.body) == 1


def test_rdna_backend_semantics_provided():
  """Docstring."""
  from ml_switcheroo.core.compiler.backends.rdna.synthesizer import RdnaBackend
  from ml_switcheroo.semantics.manager import SemanticsManager

  sem = SemanticsManager()
  backend = RdnaBackend(semantics=sem)
  assert backend.synthesizer.semantics is sem


class MockAllocator:
  """Docstring."""

  def get_vector_register(self, var_name):
    """Docstring."""
    from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaVGPR

    return RdnaVGPR(0)

  def get_scalar_register(self, var_name):
    """Docstring."""
    from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaSGPR

    return RdnaSGPR(0)

  def allocate_vector_temp(self):
    """Docstring."""
    from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaVGPR

    return RdnaVGPR(1)

  def allocate_scalar_temp(self):
    """Docstring."""
    from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaSGPR

    return RdnaSGPR(1)


def test_rdna_emit_relu_mock_fail():
  """Docstring."""
  import pytest

  from ml_switcheroo.core.compiler.backends.rdna.macros import expand_relu

  with pytest.raises(AttributeError):
    expand_relu(allocator="not_an_allocator", node_id="n1", metadata={})


def test_rdna_macros_all():
  """Docstring."""
  from ml_switcheroo.core.compiler.backends.rdna.macros import (
    expand_adam,
    expand_conv2d,
    expand_conv3d,
    expand_conv_general_dilated,
    expand_dropout,
    expand_flatten,
    expand_l,
    expand_linear,
    expand_relu,
    expand_reshape,
    expand_transpose,
    expand_variable,
  )

  alloc = MockAllocator()
  assert len(expand_conv2d(alloc, "n1", {"k": 3})) > 0
  assert len(expand_linear(alloc, "n1", {"in_features": 3})) > 0
  assert len(expand_relu(alloc, "n1", {})) > 0
  assert len(expand_flatten(alloc, "n1", {})) > 0
  assert len(expand_reshape(alloc, "n1", {})) > 0
  assert len(expand_conv3d(alloc, "n1", {"k": 3})) > 0
  assert len(expand_dropout(alloc, "n1", {})) > 0
  assert len(expand_variable(alloc, "n1", {})) > 0
  assert len(expand_transpose(alloc, "n1", {})) > 0
  assert len(expand_conv_general_dilated(alloc, "n1", {})) > 0
  assert len(expand_adam(alloc, "n1", {})) > 0
  assert len(expand_l(alloc, "n1", {})) > 0


def test_rdna_macros_linear_bias():
  """Docstring."""
  from ml_switcheroo.core.compiler.backends.rdna.macros import expand_linear

  alloc = MockAllocator()
  assert len(expand_linear(alloc, "n1", {"in_features": 3, "bias": True})) > 0
