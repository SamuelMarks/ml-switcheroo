"""Test suite for the Sass Synth Extra module."""

import pytest
import typing
from pathlib import Path
from ml_switcheroo.core.graph import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.core.compiler.backends.sass.synthesizer import SassSynthesizer, RegisterAllocator
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

  alloc = RegisterAllocator()
  nodes: list[typing.Any] = expand_mean(alloc, "n1", {"elements": 10})
  assert any(n.opcode == "FADD" for n in nodes if hasattr(n, "opcode"))
  assert any(n.opcode == "FMUL" for n in nodes if hasattr(n, "opcode"))


def test_sass_macro_relu() -> None:
  """Verifies the behavior of SASS relu macro expansion."""
  from ml_switcheroo.core.compiler.backends.sass.macros import expand_relu

  alloc = RegisterAllocator()
  nodes: list[typing.Any] = expand_relu(alloc, "n1", {})
  assert any(n.opcode == "FMAX" for n in nodes if hasattr(n, "opcode"))


def test_sass_macro_flatten() -> None:
  """Verifies the behavior of SASS flatten macro expansion."""
  from ml_switcheroo.core.compiler.backends.sass.macros import expand_flatten

  alloc = RegisterAllocator()
  nodes: list[typing.Any] = expand_flatten(alloc, "n1", {})
  assert any(n.opcode == "MOV" for n in nodes if hasattr(n, "opcode"))


def test_sass_macro_reshape() -> None:
  """Verifies the behavior of SASS reshape macro expansion."""
  from ml_switcheroo.core.compiler.backends.sass.macros import expand_reshape

  alloc = RegisterAllocator()
  nodes: list[typing.Any] = expand_reshape(alloc, "n1", {})
  assert any(n.opcode == "MOV" for n in nodes if hasattr(n, "opcode"))


def test_sass_macro_conv3d() -> None:
  """Verifies the behavior of SASS conv3d macro expansion."""
  from ml_switcheroo.core.compiler.backends.sass.macros import expand_conv3d

  alloc = RegisterAllocator()
  nodes: list[typing.Any] = expand_conv3d(alloc, "n1", {"k": 3})
  assert any(n.opcode == "FFMA" for n in nodes if hasattr(n, "opcode"))
  assert any(n.opcode == "IMAD" for n in nodes if hasattr(n, "opcode"))


def test_sass_macros_coverage() -> None:
  """Test coverage for sass macros."""
  from ml_switcheroo.core.compiler.backends.sass.macros import (
    expand_variable,
    expand_transpose,
    expand_conv_general_dilated,
    expand_adam,
    expand_l,
  )

  allocator: typing.Any = None
  assert len(expand_variable(allocator, "n", {})) == 2
  assert len(expand_transpose(allocator, "n", {})) == 2
  assert len(expand_conv_general_dilated(allocator, "n", {})) == 2
  assert len(expand_adam(allocator, "n", {})) == 2
  assert len(expand_l(allocator, "n", {})) == 2


def test_sass_synth_suffix_macro() -> None:
  """Test coverage for sass synthesizer suffix matching."""
  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("sass.l", {})
  synth = SassSynthesizer(mock_semantics)
  n1 = LogicalNode(id="n1", kind="dummy")
  n2 = LogicalNode(id="n2", kind="sass.l")
  edge = LogicalEdge(source="n1", target="n2")
  g = LogicalGraph(nodes=[n1, n2], edges=[edge])
  res: list[typing.Any] = synth.from_graph(g)
  assert len(res) >= 1


def test_sass_synth_liveness_existing() -> None:
  # Hit 130->132
  """Test sass synth liveness existing."""
  alloc = RegisterAllocator()
  g = LogicalGraph("Test")
  g.edges.append(LogicalEdge(source="s", target="t1"))
  g.edges.append(LogicalEdge(source="s", target="t2"))
  alloc.build_liveness(g)
  assert alloc._liveness_map["s"] == 2


def test_sass_synth_record_usage_missing() -> None:
  # Hit 141->exit (var_name not in _liveness_map)
  """Test sass synth record usage missing."""
  alloc = RegisterAllocator()
  alloc.record_usage("missing")  # should not throw


def test_sass_synth_record_usage_not_zero() -> None:
  # Hit 143->exit
  """Test sass synth record usage not zero."""
  alloc = RegisterAllocator()
  alloc._liveness_map["x"] = 2
  alloc.record_usage("x")
  assert alloc._liveness_map["x"] == 1


def test_sass_synth_macros_missing(monkeypatch: pytest.MonkeyPatch) -> None:
  # Hit 173->exit
  """Test sass synth macros missing."""
  import os

  monkeypatch.setattr(os.path, "exists", lambda x: False)
  synth = SassSynthesizer(None)  # type: ignore
  assert synth.macro_registry == {}


def test_sass_synth_macros_attr_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  # Hit 178->177
  """Test sass synth macros attr missing."""
  import os
  import json

  f = tmp_path / "macros.json"
  f.write_text(json.dumps({"test_op": "missing_macro_func"}))

  def mock_dirname(p: str) -> str:
    """Mock dirname."""
    return str(tmp_path)

  def mock_exists(p: str) -> bool:
    """Mock exists."""
    return p == str(f)

  monkeypatch.setattr(os.path, "dirname", mock_dirname)
  monkeypatch.setattr(os.path, "exists", mock_exists)

  synth = SassSynthesizer(None)  # type: ignore
  assert synth.macro_registry == {}


def test_sass_synth_existing_input_map() -> None:
  # Hit 218->220
  """Test sass synth existing input map."""
  synth = SassSynthesizer(None)  # type: ignore
  g = LogicalGraph("Test")
  g.edges.append(LogicalEdge(source="s", target="t"))
  g.edges.append(LogicalEdge(source="s2", target="t"))
  synth.from_graph(g)


def test_sass_synth_output_no_sources() -> None:
  # Hit 234->238
  """Test sass synth output no sources."""
  synth = SassSynthesizer(None)  # type: ignore
  g = LogicalGraph("Test")
  g.nodes.append(LogicalNode(id="out", kind="Output"))
  synth.from_graph(g)


def test_sass_synth_abstract_id_none() -> None:
  # Hit 278->281
  """Test sass synth abstract id none."""

  class FakeSemantics:
    """Fake semantics."""

    def get_definition(self, k: str) -> tuple[str, dict[str, typing.Any]]:
      """Get definition."""
      return ("", {})

  synth = SassSynthesizer(FakeSemantics())  # type: ignore
  g = LogicalGraph("Test")
  g.nodes.append(LogicalNode(id="n", kind="not_found"))
  res: list[typing.Any] = synth.from_graph(g)
  assert "Unmapped Op:" in res[0].text


def test_sass_synth_to_cst_other_node() -> None:
  # Hit 334->343
  """Test sass synth to cst other node."""
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment

  synth = SassSynthesizer(None)  # type: ignore
  mod: typing.Any = synth.to_python([SassComment("test comment")])
  assert len(mod.body) == 0


def test_sass_synth_to_cst_label() -> None:
  """Test sass synth to cst label."""
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassLabel

  synth = SassSynthesizer(None)  # type: ignore
  mod: typing.Any = synth.to_python([SassLabel("lbl")])
  assert len(mod.body) == 1


def test_sass_synth_to_cst_comment_no_begin_end() -> None:
  # Hit 334->343
  """Test sass synth to cst comment no begin end."""
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassComment

  synth = SassSynthesizer(None)  # type: ignore
  mod: typing.Any = synth.to_python([SassComment("just a regular comment")])
  assert len(mod.body) == 0


def test_sass_synth_instruction_dest_not_register() -> None:
  # Hit 408->412
  """Test sass synth instruction dest not register."""
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassInstruction, SassImmediate

  synth = SassSynthesizer(None)  # type: ignore
  # create instruction with immediate as dest
  inst = SassInstruction(opcode="OP", operands=[SassImmediate("target_var"), SassImmediate("2")])  # type: ignore
  stmt: typing.Any = synth._convert_instruction_to_py(inst)
  import libcst as cst

  code: str = cst.Module(body=[stmt]).code
  assert "sass.OP" in code


def test_sass_synth_to_cst_other_non_label_non_comment() -> None:
  # Hit 334->343
  """Test sass synth to cst other non label non comment."""
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassNode

  class CustomNode(SassNode):
    """Custom node."""

    pass

  synth = SassSynthesizer(None)  # type: ignore
  mod: typing.Any = synth.to_python([CustomNode()])
  assert len(mod.body) == 0


def test_sass_allocator_overflow() -> None:
  """Docstring."""
  allocator = RegisterAllocator()
  # It has 255 registers. Allocate them all
  with pytest.raises(ValueError):
    for i in range(256):
      allocator.allocate_temp()


def test_sass_synth_float_immediate() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassImmediate
  import libcst as cst

  synth = SassSynthesizer(None)  # type: ignore
  res: typing.Any = synth._convert_operand_to_py(SassImmediate(value=3.14))  # type: ignore
  assert isinstance(res, cst.Float)
  assert res.value == "3.14"


def test_sass_synth_expr_statement() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassInstruction, SassImmediate
  import libcst as cst

  synth = SassSynthesizer(None)  # type: ignore
  # Give it an instruction with operands, but where dest is not a Register
  # So it hits the else branch "Expression Statement"
  inst = SassInstruction(opcode="MOV", operands=[SassImmediate(value=0), SassImmediate(value=1)])  # type: ignore
  stmt: typing.Any = synth._convert_instruction_to_py(inst)
  assert isinstance(stmt, cst.SimpleStatementLine)
  assert isinstance(stmt.body[0], cst.Expr)


def test_sass_synth_no_operands() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassInstruction
  import libcst as cst

  synth = SassSynthesizer(None)  # type: ignore
  inst = SassInstruction(opcode="NOP", operands=[])
  stmt: typing.Any = synth._convert_instruction_to_py(inst)
  assert isinstance(stmt, cst.SimpleStatementLine)
  assert isinstance(stmt.body[0], cst.Expr)


def test_sass_synth_nop() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassInstruction, SassImmediate
  import libcst as cst

  synth = SassSynthesizer(None)  # type: ignore
  inst = SassInstruction(opcode="NOP", operands=[SassImmediate(value=0)])  # type: ignore
  stmt: typing.Any = synth._convert_instruction_to_py(inst)
  assert isinstance(stmt, cst.SimpleStatementLine)
  assert isinstance(stmt.body[0], cst.Expr)


def test_sass_synth_hex_immediate() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.sass.cst import SassImmediate
  import libcst as cst

  synth = SassSynthesizer(None)  # type: ignore
  # Hex immediate
  res: typing.Any = synth._convert_operand_to_py(SassImmediate(value=10, is_hex=True))  # type: ignore
  assert isinstance(res, cst.Integer)
  assert res.value == hex(10)


def test_sass_synth_output_with_sources() -> None:
  """Docstring."""
  synth = SassSynthesizer(None)  # type: ignore
  g = LogicalGraph("Test")
  g.nodes.append(LogicalNode(id="src_node", kind="Input"))
  g.nodes.append(LogicalNode(id="out", kind="Output"))
  g.edges.append(LogicalEdge(source="src_node", target="out"))
  res: list[typing.Any] = synth.from_graph(g)
  assert len(res) > 0
