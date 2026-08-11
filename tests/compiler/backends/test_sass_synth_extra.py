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


def test_sass_macro_mean() -> None:
  """Verifies the behavior of SASS mean macro expansion."""
  from ml_switcheroo.core.compiler.backends.sass.macros import expand_mean
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator

  alloc = RegisterAllocator()
  nodes = expand_mean(alloc, "n1", {"elements": 10})
  assert any(n.opcode == "FADD" for n in nodes if hasattr(n, "opcode"))
  assert any(n.opcode == "FMUL" for n in nodes if hasattr(n, "opcode"))


def test_sass_macro_relu() -> None:
  """Verifies the behavior of SASS relu macro expansion."""
  from ml_switcheroo.core.compiler.backends.sass.macros import expand_relu
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator

  alloc = RegisterAllocator()
  nodes = expand_relu(alloc, "n1", {})
  assert any(n.opcode == "FMAX" for n in nodes if hasattr(n, "opcode"))


def test_sass_macro_flatten() -> None:
  """Verifies the behavior of SASS flatten macro expansion."""
  from ml_switcheroo.core.compiler.backends.sass.macros import expand_flatten
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator

  alloc = RegisterAllocator()
  nodes = expand_flatten(alloc, "n1", {})
  assert any(n.opcode == "MOV" for n in nodes if hasattr(n, "opcode"))


def test_sass_macro_reshape() -> None:
  """Verifies the behavior of SASS reshape macro expansion."""
  from ml_switcheroo.core.compiler.backends.sass.macros import expand_reshape
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator

  alloc = RegisterAllocator()
  nodes = expand_reshape(alloc, "n1", {})
  assert any(n.opcode == "MOV" for n in nodes if hasattr(n, "opcode"))


def test_sass_macro_conv3d() -> None:
  """Verifies the behavior of SASS conv3d macro expansion."""
  from ml_switcheroo.core.compiler.backends.sass.macros import expand_conv3d
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator

  alloc = RegisterAllocator()
  nodes = expand_conv3d(alloc, "n1", {"k": 3})
  assert any(n.opcode == "FFMA" for n in nodes if hasattr(n, "opcode"))
  assert any(n.opcode == "IMAD" for n in nodes if hasattr(n, "opcode"))


def test_sass_macros_coverage():
  """Test coverage for sass macros."""
  from ml_switcheroo.core.compiler.backends.sass.macros import (
    expand_variable,
    expand_transpose,
    expand_conv_general_dilated,
    expand_adam,
    expand_l,
  )

  allocator = None
  assert len(expand_variable(allocator, "n", {})) == 2
  assert len(expand_transpose(allocator, "n", {})) == 2
  assert len(expand_conv_general_dilated(allocator, "n", {})) == 2
  assert len(expand_adam(allocator, "n", {})) == 2
  assert len(expand_l(allocator, "n", {})) == 2


def test_sass_synth_suffix_macro():
  """Test coverage for sass synthesizer suffix matching."""
  from ml_switcheroo.core.graph import LogicalGraph, LogicalNode, LogicalEdge

  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("sass.l", {})
  synth = SassSynthesizer(mock_semantics)
  n1 = LogicalNode(id="n1", kind="dummy")
  n2 = LogicalNode(id="n2", kind="sass.l")
  edge = LogicalEdge(source="n1", target="n2")
  g = LogicalGraph(nodes=[n1, n2], edges=[edge])
  res = synth.from_graph(g)
  assert len(res) >= 1


def test_sass_synth_liveness_existing():
  # Hit 130->132
  """Test sass synth liveness existing."""
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator
  from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalEdge

  alloc = RegisterAllocator()
  g = LogicalGraph("Test")
  g.edges.append(LogicalEdge(source="s", target="t1"))
  g.edges.append(LogicalEdge(source="s", target="t2"))
  alloc.build_liveness(g)
  assert alloc._liveness_map["s"] == 2


def test_sass_synth_record_usage_missing():
  # Hit 141->exit (var_name not in _liveness_map)
  """Test sass synth record usage missing."""
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator

  alloc = RegisterAllocator()
  alloc.record_usage("missing")  # should not throw


def test_sass_synth_record_usage_not_zero():
  # Hit 143->exit
  """Test sass synth record usage not zero."""
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import RegisterAllocator

  alloc = RegisterAllocator()
  alloc._liveness_map["x"] = 2
  alloc.record_usage("x")
  assert alloc._liveness_map["x"] == 1


def test_sass_synth_macros_missing(monkeypatch):
  # Hit 173->exit
  """Test sass synth macros missing."""
  import os
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import SassSynthesizer

  monkeypatch.setattr(os.path, "exists", lambda x: False)
  synth = SassSynthesizer(None)
  assert synth.macro_registry == {}


def test_sass_synth_macros_attr_missing(monkeypatch, tmp_path):
  # Hit 178->177
  """Test sass synth macros attr missing."""
  import os
  import json
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import SassSynthesizer

  f = tmp_path / "macros.json"
  f.write_text(json.dumps({"test_op": "missing_macro_func"}))

  def mock_dirname(p):
    """Mock dirname."""
    return str(tmp_path)

  def mock_exists(p):
    """Mock exists."""
    return p == str(f)

  monkeypatch.setattr(os.path, "dirname", mock_dirname)
  monkeypatch.setattr(os.path, "exists", mock_exists)

  synth = SassSynthesizer(None)
  assert synth.macro_registry == {}


def test_sass_synth_existing_input_map():
  # Hit 218->220
  """Test sass synth existing input map."""
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import SassSynthesizer
  from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalEdge

  synth = SassSynthesizer(None)
  g = LogicalGraph("Test")
  g.edges.append(LogicalEdge(source="s", target="t"))
  g.edges.append(LogicalEdge(source="s2", target="t"))
  synth.from_graph(g)


def test_sass_synth_output_no_sources():
  # Hit 234->238
  """Test sass synth output no sources."""
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import SassSynthesizer
  from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode

  synth = SassSynthesizer(None)
  g = LogicalGraph("Test")
  g.nodes.append(LogicalNode(id="out", kind="Output"))
  synth.from_graph(g)


def test_sass_synth_abstract_id_none():
  # Hit 278->281
  """Test sass synth abstract id none."""
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import SassSynthesizer
  from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode

  class FakeSemantics:
    """Fake semantics."""

    def get_definition(self, k):
      """Get definition."""
      return ("", {})

  synth = SassSynthesizer(FakeSemantics())
  g = LogicalGraph("Test")
  g.nodes.append(LogicalNode(id="n", kind="not_found"))
  res = synth.from_graph(g)
  assert "Unmapped Op:" in res[0].text


def test_sass_synth_to_cst_other_node():
  # Hit 334->343
  """Test sass synth to cst other node."""
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import SassSynthesizer
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment

  synth = SassSynthesizer(None)
  mod = synth.to_python([SassComment("test comment")])
  assert len(mod.body) == 0


def test_sass_synth_to_cst_label():
  """Test sass synth to cst label."""
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import SassSynthesizer
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassLabel

  synth = SassSynthesizer(None)
  mod = synth.to_python([SassLabel("lbl")])
  assert len(mod.body) == 1


def test_sass_synth_to_cst_comment_no_begin_end():
  # Hit 334->343
  """Test sass synth to cst comment no begin end."""
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import SassSynthesizer
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment

  synth = SassSynthesizer(None)
  mod = synth.to_python([SassComment("just a regular comment")])
  assert len(mod.body) == 0


def test_sass_synth_instruction_dest_not_register():
  # Hit 408->412
  """Test sass synth instruction dest not register."""
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import SassSynthesizer
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassInstruction, SassImmediate

  synth = SassSynthesizer(None)
  # create instruction with immediate as dest
  inst = SassInstruction(opcode="OP", operands=[SassImmediate("target_var"), SassImmediate("2")])
  stmt = synth._convert_instruction_to_py(inst)
  import libcst as cst

  code = cst.Module(body=[stmt]).code
  assert "sass.OP" in code


def test_sass_synth_to_cst_other_non_label_non_comment():
  # Hit 334->343
  """Test sass synth to cst other non label non comment."""
  from ml_switcheroo.core.compiler.backends.sass.synthesizer import SassSynthesizer
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassNode

  class CustomNode(SassNode):
    """Custom node."""

    pass

  synth = SassSynthesizer(None)
  mod = synth.to_python([CustomNode()])
  assert len(mod.body) == 0
