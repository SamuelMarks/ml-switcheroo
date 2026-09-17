"""Test module."""

from typing import Dict, List, Optional
from pathlib import Path
from unittest.mock import patch

import libcst as cst
import pytest

from ml_switcheroo.core.compiler.backends.rdna.synthesizer import (
  MAX_SGPR,
  MAX_VGPR,
  RdnaBackend,
  RdnaSynthesizer,
  RegisterAllocator,
)
from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaLabel,
  RdnaImmediate,
  RdnaInstruction,
  RdnaNode,
  RdnaVGPR,
  RdnaSGPR,
)
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode


class MockSemanticsManager:
  """Docstring."""

  def get_definition(self, kind: str) -> Optional[List[str]]:
    """Docstring."""
    if kind == "known_op":
      return ["add"]
    elif kind == "macro_op":
      return ["relu"]
    elif kind == "macro_suffix_op":
      return ["some.relu"]
    return None

  def resolve_variant(self, abstract_id: str, backend: str) -> Optional[Dict[str, str]]:
    """Docstring."""
    if abstract_id == "add":
      return {"api": "v_add_f32"}
    return None


def test_register_allocator() -> None:
  """Docstring."""
  allocator: RegisterAllocator = RegisterAllocator()

  v1: RdnaVGPR = allocator.get_vector_register("x")
  v2: RdnaVGPR = allocator.get_vector_register("x")
  assert v1.index == v2.index

  s1: RdnaSGPR = allocator.get_scalar_register("y")
  s2: RdnaSGPR = allocator.get_scalar_register("y")
  assert s1.index == s2.index

  v_temp: RdnaVGPR = allocator.allocate_vector_temp()
  assert isinstance(v_temp, RdnaVGPR)
  assert v_temp.index != v1.index

  s_temp: RdnaSGPR = allocator.allocate_scalar_temp()
  assert isinstance(s_temp, RdnaSGPR)
  assert s_temp.index != s1.index

  allocator.reset()
  assert allocator._next_vgpr == 0
  assert allocator._next_sgpr == 0


def test_register_allocator_overflow() -> None:
  """Docstring."""
  allocator: RegisterAllocator = RegisterAllocator()
  allocator._next_vgpr = MAX_VGPR
  with pytest.raises(ValueError, match="RdnaVGPR overflow"):
    allocator.get_vector_register("new_var")

  allocator._next_sgpr = MAX_SGPR
  with pytest.raises(ValueError, match="RdnaSGPR overflow"):
    allocator.get_scalar_register("new_var_s")


def test_synthesizer_from_graph() -> None:
  """Docstring."""
  semantics: MockSemanticsManager = MockSemanticsManager()
  synthesizer: RdnaSynthesizer = RdnaSynthesizer(semantics)

  n_in: LogicalNode = LogicalNode(id="in1", op_type="Input", attributes={"name": "input_x"})
  n_op1: LogicalNode = LogicalNode(id="op1", op_type="known_op", inputs=["in1"])
  n_op2: LogicalNode = LogicalNode(id="op2", op_type="unknown_op", inputs=["op1"])
  n_out: LogicalNode = LogicalNode(id="out1", op_type="Output", inputs=["op2"])
  n_macro: LogicalNode = LogicalNode(id="op_macro", op_type="macro_op")
  n_macro_suffix: LogicalNode = LogicalNode(id="op_macro_s", op_type="macro_suffix_op")

  graph: LogicalGraph = LogicalGraph(
    nodes={
      "in1": n_in,
      "op1": n_op1,
      "op2": n_op2,
      "out1": n_out,
      "op_macro": n_macro,
      "op_macro_s": n_macro_suffix,
    }
  )

  # We also test a node that resolves to an abstract_id with a variant, but no 'api'
  semantics.resolve_variant = lambda aid, b: {"api": "v_add_f32"} if aid == "add" else {}

  nodes: List[RdnaNode] = synthesizer.from_graph(graph)
  assert len(nodes) > 0


def test_synthesizer_to_python() -> None:
  """Docstring."""
  semantics: MockSemanticsManager = MockSemanticsManager()
  synthesizer: RdnaSynthesizer = RdnaSynthesizer(semantics)

  inst1: RdnaInstruction = RdnaInstruction(
    opcode="v_add_f32", operands=[RdnaVGPR(index=0), RdnaVGPR(index=1), RdnaVGPR(index=2)]
  )
  inst2: RdnaInstruction = RdnaInstruction(opcode="store_dword", operands=[RdnaVGPR(index=0), RdnaImmediate(value=4)])
  inst3: RdnaInstruction = RdnaInstruction(opcode="s_waitcnt", operands=[])
  inst4: RdnaInstruction = RdnaInstruction(opcode="s_cbranch_vccnz", operands=[RdnaLabel(name="L1")])
  inst5: RdnaInstruction = RdnaInstruction(
    opcode="v_add_f32", operands=[RdnaVGPR(index=0), RdnaImmediate(value=1.5, is_hex=False)]
  )
  inst6: RdnaInstruction = RdnaInstruction(
    opcode="v_add_f32", operands=[RdnaVGPR(index=0), RdnaImmediate(value=0x10, is_hex=True)]
  )
  inst7: RdnaInstruction = RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), "some_string"])
  inst8: RdnaInstruction = RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), "array[0]"])
  inst9: RdnaInstruction = RdnaInstruction(opcode="v_add_f32", operands=["bad_dest", "array[0]"])  # Hit 295->301
  label1: RdnaLabel = RdnaLabel(name="L1")
  comment: RdnaComment = RdnaComment(text="; Comment")

  cst_module: cst.Module = synthesizer.to_python(
    [inst1, inst2, inst3, inst4, inst5, inst6, inst7, inst8, inst9, label1, comment]
  )
  assert isinstance(cst_module, cst.Module)


def test_rdna_backend() -> None:
  """Docstring."""
  semantics: MockSemanticsManager = MockSemanticsManager()
  backend: RdnaBackend = RdnaBackend(semantics)

  graph: LogicalGraph = LogicalGraph()
  n_in: LogicalNode = LogicalNode(id="in1", op_type="Input")
  graph.add_node(n_in)

  result: str = backend.compile(graph)
  assert "RDNA Code Generation Initialized" in result


def test_rdna_backend_default_semantics() -> None:
  """Docstring."""
  backend: RdnaBackend = RdnaBackend()
  assert backend.synthesizer.semantics is not None


# --- Merged from test_rdna_backend_synthesizer_extra.py ---


class DummySemantics:
  """Docstring."""

  def resolve_variant(self, abstract_id: str, flavor: str) -> Optional[dict]:
    """Docstring."""
    return None

  def get_definition(self, kind: str) -> Optional[dict]:
    """Docstring."""
    return None


def test_rdna_synthesizer_branches() -> None:
  """Docstring."""
  # 1. Test when macros.json does not exist
  with patch("os.path.exists", return_value=False):
    synth: RdnaSynthesizer = RdnaSynthesizer(semantics=DummySemantics())
    assert not getattr(synth, "macro_registry")

  # 2. Test when macros.json exists but is empty
  with patch("os.path.exists", return_value=True):
    with patch("builtins.open", __import__("unittest").mock.mock_open(read_data="{}")):
      synth2: RdnaSynthesizer = RdnaSynthesizer(semantics=DummySemantics())
      assert not getattr(synth2, "macro_registry")

  # 3. Test multiple edges to same target (174->176)
  nodes = [
    LogicalNode(id="in1", op_type="Input"),
    LogicalNode(id="in2", op_type="Input"),
    LogicalNode(id="add", op_type="Add"),
    LogicalNode(id="out", op_type="Output"),  # Empty output sources?
  ]
  edges = [
    LogicalEdge(source="in1", target="add"),
    LogicalEdge(source="in2", target="add"),  # multiple edges
  ]
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=edges)
  synth3: RdnaSynthesizer = RdnaSynthesizer(semantics=DummySemantics())
  synth3.from_graph(graph)

  # 4. Test when output has no sources (188->178)
  nodes2 = [LogicalNode(id="out2", op_type="Output")]
  edges2 = []
  graph2: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes2}, edges=edges2)
  synth4: RdnaSynthesizer = RdnaSynthesizer(semantics=DummySemantics())
  synth4.from_graph(graph2)


def test_synthesizer_init(tmp_path: Path) -> None:
  """Docstring."""
  macros_json: Path = tmp_path / "macros.json"
  macros_json.write_text('{"Conv2d": "expand_conv2d", "UnknownOp": "expand_unknown"}')

  semantics: MockSemanticsManager = MockSemanticsManager()

  with patch("ml_switcheroo.core.compiler.backends.rdna.synthesizer.os.path.dirname", return_value=str(tmp_path)):
    with patch("ml_switcheroo.core.compiler.backends.rdna.synthesizer.os.path.exists", return_value=True):
      synthesizer: RdnaSynthesizer = RdnaSynthesizer(semantics)
      assert "Conv2d" in synthesizer.macro_registry
      assert "UnknownOp" not in synthesizer.macro_registry

  with patch("ml_switcheroo.core.compiler.backends.rdna.synthesizer.os.path.exists", return_value=False):
    synth2: RdnaSynthesizer = RdnaSynthesizer(semantics)
    assert "Conv2d" not in synth2.macro_registry


def test_synthesizer_init_yaml(tmp_path: Path) -> None:
  """Tests RDNA synthesizer initialization when macros.yaml is present."""
  macros_yaml: Path = tmp_path / "macros.yaml"
  macros_yaml.write_text("Conv2d: expand_conv2d\nUnknownOp: expand_unknown\n")

  semantics: MockSemanticsManager = MockSemanticsManager()
  with patch("ml_switcheroo.core.compiler.backends.rdna.synthesizer.os.path.dirname", return_value=str(tmp_path)):
    synth: RdnaSynthesizer = RdnaSynthesizer(semantics)
    assert "Conv2d" in synth.macro_registry
    assert "UnknownOp" not in synth.macro_registry


def test_synthesizer_init_filenotfound(tmp_path: Path) -> None:
  """Tests RDNA synthesizer initialization when files exist in os.path.exists check but fail on open."""
  semantics: MockSemanticsManager = MockSemanticsManager()
  with patch("ml_switcheroo.core.compiler.backends.rdna.synthesizer.os.path.dirname", return_value=str(tmp_path)):
    with patch("ml_switcheroo.core.compiler.backends.rdna.synthesizer.os.path.exists", return_value=True):
      synth: RdnaSynthesizer = RdnaSynthesizer(semantics)
      assert synth.macro_registry == {}


def test_unmapped_op_and_comment() -> None:
  """Docstring."""
  # Test Unmapped Op and RdnaComment branches
  nodes = [LogicalNode(id="unmapped1", op_type="TotallyUnknownOp")]
  nodes.append(LogicalNode(id="n2", op_type="", attributes={}))  # Hit 211->214
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in nodes}, edges=[])

  synth: RdnaSynthesizer = RdnaSynthesizer(semantics=DummySemantics())
  cst_mod: list = synth.from_graph(graph)
  assert cst_mod is not None
