"""Test module."""

from typing import Dict, List, Optional
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
  RdnaImmediate,
  RdnaInstruction,
  RdnaLabel,
  RdnaNode,
  RdnaSGPR,
  RdnaVGPR,
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

  graph: LogicalGraph = LogicalGraph()
  n_in: LogicalNode = LogicalNode(id="in1", kind="Input", metadata={"name": "input_x"})
  n_out: LogicalNode = LogicalNode(id="out1", kind="Output")
  n_op1: LogicalNode = LogicalNode(id="op1", kind="known_op")
  n_op2: LogicalNode = LogicalNode(id="op2", kind="unknown_op")
  n_macro: LogicalNode = LogicalNode(id="op_macro", kind="macro_op")
  n_macro_suffix: LogicalNode = LogicalNode(id="op_macro_s", kind="macro_suffix_op")

  graph.nodes.extend([n_in, n_out, n_op1, n_op2, n_macro, n_macro_suffix])
  graph.edges.append(LogicalEdge(source="in1", target="op1"))
  graph.edges.append(LogicalEdge(source="op1", target="op2"))
  graph.edges.append(LogicalEdge(source="op2", target="out1"))

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
  label1: RdnaLabel = RdnaLabel(name="L1")

  cst_module: cst.Module = synthesizer.to_python([inst1, inst2, inst3, inst4, inst5, inst6, inst7, inst8, label1])
  assert isinstance(cst_module, cst.Module)


def test_rdna_backend() -> None:
  """Docstring."""
  semantics: MockSemanticsManager = MockSemanticsManager()
  backend: RdnaBackend = RdnaBackend(semantics)

  graph: LogicalGraph = LogicalGraph()
  n_in: LogicalNode = LogicalNode(id="in1", kind="Input")
  graph.nodes.append(n_in)

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
    LogicalNode(id="in1", kind="Input"),
    LogicalNode(id="in2", kind="Input"),
    LogicalNode(id="add", kind="Add"),
    LogicalNode(id="out", kind="Output"),  # Empty output sources?
  ]
  edges = [
    LogicalEdge(source="in1", target="add"),
    LogicalEdge(source="in2", target="add"),  # multiple edges
  ]
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=edges)
  synth3: RdnaSynthesizer = RdnaSynthesizer(semantics=DummySemantics())
  synth3.from_graph(graph)

  # 4. Test when output has no sources (188->178)
  nodes2 = [LogicalNode(id="out2", kind="Output")]
  edges2 = []
  graph2: LogicalGraph = LogicalGraph(nodes=nodes2, edges=edges2)
  synth4: RdnaSynthesizer = RdnaSynthesizer(semantics=DummySemantics())
  synth4.from_graph(graph2)


def test_unmapped_op_and_comment() -> None:
  """Docstring."""
  # Test Unmapped Op and RdnaComment branches
  nodes = [LogicalNode(id="unmapped1", kind="TotallyUnknownOp")]
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=[])

  synth: RdnaSynthesizer = RdnaSynthesizer(semantics=DummySemantics())
  cst_mod: list = synth.from_graph(graph)
  assert cst_mod is not None
