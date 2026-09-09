"""Test suite for the Sass Synth Extra module."""

import typing
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import RegisterAllocator, NvidiaSassSynthesizer
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.graph import LogicalEdge, LogicalGraph, LogicalNode  # noqa: F811
from ml_switcheroo.semantics.manager import SemanticsManager


def test_nvidia_sass_synth_invalid_opcode() -> None:
  """Verifies the behavior of NVIDIA_SASS synth invalid opcode."""
  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("BadOp", {})
  mock_semantics.resolve_variant.return_value = {"api": "bad op code!"}
  synth = NvidiaSassSynthesizer(mock_semantics)
  g = LogicalGraph(nodes=[LogicalNode(id="n1", kind="BadOp")])
  with pytest.raises(ValueError, match="Invalid NVIDIA_SASS opcode"):
    synth.from_graph(g)


def test_nvidia_sass_synth_liveness_tracking() -> None:
  """Verifies the behavior of NVIDIA_SASS synth liveness tracking."""
  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("Add", {})
  mock_semantics.resolve_variant.return_value = {"api": "FADD"}
  synth = NvidiaSassSynthesizer(mock_semantics)
  g = LogicalGraph()
  g.nodes = [LogicalNode(id="n1", kind="Input"), LogicalNode(id="n2", kind="Add")]
  g.edges = [LogicalEdge(source="n1", target="n2")]
  assert synth.allocator._liveness_map == {}
  synth.from_graph(g)
  assert synth.allocator._liveness_map["n1"] == 0
  assert len(synth.allocator._free_pool) == 254


def test_nvidia_sass_macro_mean() -> None:
  """Verifies the behavior of NVIDIA_SASS mean macro expansion."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_mean

  alloc = RegisterAllocator()
  nodes: list[typing.Any] = expand_mean(alloc, "n1", {"elements": 10})
  assert any(n.opcode == "FADD" for n in nodes if hasattr(n, "opcode"))
  assert any(n.opcode == "FMUL" for n in nodes if hasattr(n, "opcode"))


def test_nvidia_sass_macro_relu() -> None:
  """Verifies the behavior of NVIDIA_SASS relu macro expansion."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_relu

  alloc = RegisterAllocator()
  nodes: list[typing.Any] = expand_relu(alloc, "n1", {})
  assert any(n.opcode == "FMAX" for n in nodes if hasattr(n, "opcode"))


def test_nvidia_sass_macro_flatten() -> None:
  """Verifies the behavior of NVIDIA_SASS flatten macro expansion."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_flatten

  alloc = RegisterAllocator()
  nodes: list[typing.Any] = expand_flatten(alloc, "n1", {})
  assert any(n.opcode == "MOV" for n in nodes if hasattr(n, "opcode"))


def test_nvidia_sass_macro_reshape() -> None:
  """Verifies the behavior of NVIDIA_SASS reshape macro expansion."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_reshape

  alloc = RegisterAllocator()
  nodes: list[typing.Any] = expand_reshape(alloc, "n1", {})
  assert any(n.opcode == "MOV" for n in nodes if hasattr(n, "opcode"))


def test_nvidia_sass_macro_conv3d() -> None:
  """Verifies the behavior of NVIDIA_SASS conv3d macro expansion."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import expand_conv3d

  alloc = RegisterAllocator()
  nodes: list[typing.Any] = expand_conv3d(alloc, "n1", {"k": 3})
  assert any(n.opcode == "FFMA" for n in nodes if hasattr(n, "opcode"))
  assert any(n.opcode == "IMAD" for n in nodes if hasattr(n, "opcode"))


def test_nvidia_sass_macros_coverage() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass.macros import (
    expand_adam,
    expand_conv_general_dilated,
    expand_l,
    expand_transpose,
    expand_variable,
  )

  allocator: typing.Any = None
  assert len(expand_variable(allocator, "n", {})) == 2
  assert len(expand_transpose(allocator, "n", {})) == 2
  assert len(expand_conv_general_dilated(allocator, "n", {})) == 2
  assert len(expand_adam(allocator, "n", {})) == 2
  assert len(expand_l(allocator, "n", {})) == 2


def test_nvidia_sass_synth_suffix_macro() -> None:
  """Docstring."""
  mock_semantics = MagicMock()
  mock_semantics.get_definition.return_value = ("nvidia_sass.l", {})
  synth = NvidiaSassSynthesizer(mock_semantics)
  n1 = LogicalNode(id="n1", kind="dummy")
  n2 = LogicalNode(id="n2", kind="nvidia_sass.l")
  edge = LogicalEdge(source="n1", target="n2")
  g = LogicalGraph(nodes=[n1, n2], edges=[edge])
  res: list[typing.Any] = synth.from_graph(g)
  assert len(res) >= 1


def test_nvidia_sass_synth_liveness_existing() -> None:
  """Docstring."""
  # Hit 130->132
  alloc = RegisterAllocator()
  g = LogicalGraph("Test")
  g.edges.append(LogicalEdge(source="s", target="t1"))
  g.edges.append(LogicalEdge(source="s", target="t2"))
  alloc.build_liveness(g)
  assert alloc._liveness_map["s"] == 2


def test_nvidia_sass_synth_record_usage_missing() -> None:
  """Docstring."""
  # Hit 141->exit (var_name not in _liveness_map)
  alloc = RegisterAllocator()
  alloc.record_usage("missing")  # should not throw


def test_nvidia_sass_synth_record_usage_not_zero() -> None:
  """Docstring."""
  # Hit 143->exit
  alloc = RegisterAllocator()
  alloc._liveness_map["x"] = 2
  alloc.record_usage("x")
  assert alloc._liveness_map["x"] == 1


def test_nvidia_sass_synth_macros_missing(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  # Hit 173->exit
  import os

  monkeypatch.setattr(os.path, "exists", lambda x: False)
  synth = NvidiaSassSynthesizer(None)  # type: ignore
  assert synth.macro_registry == {}


def test_nvidia_sass_synth_macros_attr_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  """Docstring."""
  # Hit 178->177
  import json
  import os

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

  synth = NvidiaSassSynthesizer(None)  # type: ignore
  assert synth.macro_registry == {}


def test_nvidia_sass_synth_existing_input_map() -> None:
  """Docstring."""
  # Hit 218->220
  synth = NvidiaSassSynthesizer(None)  # type: ignore
  g = LogicalGraph("Test")
  g.edges.append(LogicalEdge(source="s", target="t"))
  g.edges.append(LogicalEdge(source="s2", target="t"))
  synth.from_graph(g)


def test_nvidia_sass_synth_output_no_sources() -> None:
  """Docstring."""
  # Hit 234->238
  synth = NvidiaSassSynthesizer(None)  # type: ignore
  g = LogicalGraph("Test")
  g.nodes.append(LogicalNode(id="out", kind="Output"))
  synth.from_graph(g)


def test_nvidia_sass_synth_abstract_id_none() -> None:
  """Docstring."""
  # Hit 278->281

  class FakeSemantics:
    """Fake semantics."""

    def get_definition(self, k: str) -> tuple[str, dict[str, typing.Any]]:
      """Get definition."""
      return ("", {})

  synth = NvidiaSassSynthesizer(FakeSemantics())  # type: ignore
  g = LogicalGraph("Test")
  g.nodes.append(LogicalNode(id="n", kind="not_found"))
  res: list[typing.Any] = synth.from_graph(g)
  assert "Unmapped Op:" in res[0].text


def test_nvidia_sass_synth_to_cst_other_node() -> None:
  """Docstring."""
  # Hit 334->343
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment

  synth = NvidiaSassSynthesizer(None)  # type: ignore
  mod: typing.Any = synth.to_python([NvidiaSassComment("test comment")])
  assert len(mod.body) == 0


def test_nvidia_sass_synth_to_cst_label() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassLabel

  synth = NvidiaSassSynthesizer(None)  # type: ignore
  mod: typing.Any = synth.to_python([NvidiaSassLabel("lbl")])
  assert len(mod.body) == 1


def test_nvidia_sass_synth_to_cst_comment_no_begin_end() -> None:
  """Docstring."""
  # Hit 334->343
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassComment

  synth = NvidiaSassSynthesizer(None)  # type: ignore
  mod: typing.Any = synth.to_python([NvidiaSassComment("just a regular comment")])
  assert len(mod.body) == 0


def test_nvidia_sass_synth_instruction_dest_not_register() -> None:
  """Docstring."""
  # Hit 408->412
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassImmediate, NvidiaSassInstruction

  synth = NvidiaSassSynthesizer(None)  # type: ignore
  # create instruction with immediate as dest
  inst = NvidiaSassInstruction(opcode="OP", operands=[NvidiaSassImmediate("target_var"), NvidiaSassImmediate("2")])  # type: ignore
  stmt: typing.Any = synth._convert_instruction_to_py(inst)
  import libcst as cst

  code: str = cst.Module(body=[stmt]).code
  assert "nvidia_sass.OP" in code


def test_nvidia_sass_synth_to_cst_other_non_label_non_comment() -> None:
  """Docstring."""
  # Hit 334->343
  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassNode

  class CustomNode(NvidiaSassNode):
    """Custom node."""

    pass

  synth = NvidiaSassSynthesizer(None)  # type: ignore
  mod: typing.Any = synth.to_python([CustomNode()])
  assert len(mod.body) == 0


def test_nvidia_sass_allocator_overflow() -> None:
  """Docstring."""
  allocator = RegisterAllocator()
  # It has 255 registers. Allocate them all
  with pytest.raises(ValueError):
    for i in range(256):
      allocator.allocate_temp()


def test_nvidia_sass_synth_float_immediate() -> None:
  """Docstring."""
  import libcst as cst

  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassImmediate

  synth = NvidiaSassSynthesizer(None)  # type: ignore
  res: typing.Any = synth._convert_operand_to_py(NvidiaSassImmediate(value=3.14))  # type: ignore
  assert isinstance(res, cst.Float)
  assert res.value == "3.14"


def test_nvidia_sass_synth_expr_statement() -> None:
  """Docstring."""
  import libcst as cst

  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassImmediate, NvidiaSassInstruction

  synth = NvidiaSassSynthesizer(None)  # type: ignore
  # Give it an instruction with operands, but where dest is not a Register
  # So it hits the else branch "Expression Statement"
  inst = NvidiaSassInstruction(opcode="MOV", operands=[NvidiaSassImmediate(value=0), NvidiaSassImmediate(value=1)])  # type: ignore
  stmt: typing.Any = synth._convert_instruction_to_py(inst)
  assert isinstance(stmt, cst.SimpleStatementLine)
  assert isinstance(stmt.body[0], cst.Expr)


def test_nvidia_sass_synth_no_operands() -> None:
  """Docstring."""
  import libcst as cst

  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassInstruction

  synth = NvidiaSassSynthesizer(None)  # type: ignore
  inst = NvidiaSassInstruction(opcode="NOP", operands=[])
  stmt: typing.Any = synth._convert_instruction_to_py(inst)
  assert isinstance(stmt, cst.SimpleStatementLine)
  assert isinstance(stmt.body[0], cst.Expr)


def test_nvidia_sass_synth_nop() -> None:
  """Docstring."""
  import libcst as cst

  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassImmediate, NvidiaSassInstruction

  synth = NvidiaSassSynthesizer(None)  # type: ignore
  inst = NvidiaSassInstruction(opcode="NOP", operands=[NvidiaSassImmediate(value=0)])  # type: ignore
  stmt: typing.Any = synth._convert_instruction_to_py(inst)
  assert isinstance(stmt, cst.SimpleStatementLine)
  assert isinstance(stmt.body[0], cst.Expr)


def test_nvidia_sass_synth_hex_immediate() -> None:
  """Docstring."""
  import libcst as cst

  from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import NvidiaSassImmediate

  synth = NvidiaSassSynthesizer(None)  # type: ignore
  # Hex immediate
  res: typing.Any = synth._convert_operand_to_py(NvidiaSassImmediate(value=10, is_hex=True))  # type: ignore
  assert isinstance(res, cst.Integer)
  assert res.value == hex(10)


def test_nvidia_sass_synth_output_with_sources() -> None:
  """Docstring."""
  synth = NvidiaSassSynthesizer(None)  # type: ignore
  g = LogicalGraph("Test")
  g.nodes.append(LogicalNode(id="src_node", kind="Input"))
  g.nodes.append(LogicalNode(id="out", kind="Output"))
  g.edges.append(LogicalEdge(source="src_node", target="out"))
  res: list[typing.Any] = synth.from_graph(g)
  assert len(res) > 0


# --- Merged from test_nvidia_sass_synth_missing.py ---


def test_nvidia_sass_synthesizer_macro_exact_match() -> None:
  """Docstring."""
  semantics = SemanticsManager()
  synth = NvidiaSassSynthesizer(semantics)

  synth.macro_registry["my_macro"] = lambda alloc, nid, meta: [NvidiaSassComment(text="mock_my_macro")]

  graph = LogicalGraph("test")
  n1 = LogicalNode("n1", "tensor")
  n2 = LogicalNode("n2", "my_macro")
  graph.nodes.extend([n1, n2])
  graph.edges.append(LogicalEdge("n1", "n2"))

  original = semantics.get_definition

  def mock_get_def(kind: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Docstring."""
    if kind == "my_macro":
      return ("my_macro", {})
    return original(kind)

  semantics.get_definition = mock_get_def  # type: ignore

  nodes: list[typing.Any] = synth.from_graph(graph)

  found = False
  for n in nodes:
    if isinstance(n, NvidiaSassComment) and n.text == "mock_my_macro":
      found = True
  assert found


def test_nvidia_sass_backend_semantics_provided() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.backends.nvidia_sass.backend import NvidiaSassBackend
  from ml_switcheroo.semantics.manager import SemanticsManager

  sem = SemanticsManager()
  backend = NvidiaSassBackend(semantics=sem)
  assert backend.synthesizer.semantics is sem
