"""Test module."""

import pytest
import libcst as cst

from ml_switcheroo.core.compiler.backends.rdna.synthesizer import (
  RegisterAllocator,
  RdnaSynthesizer,
  RdnaBackend,
  MAX_VGPR,
  MAX_SGPR,
)
from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaVGPR,
  RdnaSGPR,
  RdnaInstruction,
  RdnaLabel,
  RdnaImmediate,
)
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge


class MockSemanticsManager:
  """Test element."""

  def get_definition(self, kind: str):
    """Test element."""
    if kind == "known_op":
      return ["add"]
    elif kind == "macro_op":
      return ["relu"]
    elif kind == "macro_suffix_op":
      return ["some.relu"]
    return None

  def resolve_variant(self, abstract_id: str, backend: str):
    """Test element."""
    if abstract_id == "add":
      return {"api": "v_add_f32"}
    return None


def test_register_allocator():
  """Test element."""
  allocator = RegisterAllocator()

  v1 = allocator.get_vector_register("x")
  v2 = allocator.get_vector_register("x")
  assert v1.index == v2.index

  s1 = allocator.get_scalar_register("y")
  s2 = allocator.get_scalar_register("y")
  assert s1.index == s2.index

  v_temp = allocator.allocate_vector_temp()
  assert isinstance(v_temp, RdnaVGPR)
  assert v_temp.index != v1.index

  s_temp = allocator.allocate_scalar_temp()
  assert isinstance(s_temp, RdnaSGPR)
  assert s_temp.index != s1.index

  allocator.reset()
  assert allocator._next_vgpr == 0
  assert allocator._next_sgpr == 0


def test_register_allocator_overflow():
  """Test element."""
  allocator = RegisterAllocator()
  allocator._next_vgpr = MAX_VGPR
  with pytest.raises(ValueError, match="RdnaVGPR overflow"):
    allocator.get_vector_register("new_var")

  allocator._next_sgpr = MAX_SGPR
  with pytest.raises(ValueError, match="RdnaSGPR overflow"):
    allocator.get_scalar_register("new_var_s")


def test_synthesizer_from_graph():
  """Test element."""
  semantics = MockSemanticsManager()
  synthesizer = RdnaSynthesizer(semantics)

  graph = LogicalGraph()
  n_in = LogicalNode(id="in1", kind="Input", metadata={"name": "input_x"})
  n_out = LogicalNode(id="out1", kind="Output")
  n_op1 = LogicalNode(id="op1", kind="known_op")
  n_op2 = LogicalNode(id="op2", kind="unknown_op")
  n_macro = LogicalNode(id="op_macro", kind="macro_op")
  n_macro_suffix = LogicalNode(id="op_macro_s", kind="macro_suffix_op")

  graph.nodes.extend([n_in, n_out, n_op1, n_op2, n_macro, n_macro_suffix])
  graph.edges.append(LogicalEdge(source="in1", target="op1"))
  graph.edges.append(LogicalEdge(source="op1", target="op2"))
  graph.edges.append(LogicalEdge(source="op2", target="out1"))

  # We also test a node that resolves to an abstract_id with a variant, but no 'api'
  semantics.resolve_variant = lambda aid, b: {"api": "v_add_f32"} if aid == "add" else {}

  nodes = synthesizer.from_graph(graph)
  assert len(nodes) > 0


def test_synthesizer_to_python():
  """Test element."""
  semantics = MockSemanticsManager()
  synthesizer = RdnaSynthesizer(semantics)

  inst1 = RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), RdnaVGPR(index=1), RdnaVGPR(index=2)])
  inst2 = RdnaInstruction(opcode="store_dword", operands=[RdnaVGPR(index=0), RdnaImmediate(value=4)])
  inst3 = RdnaInstruction(opcode="s_waitcnt", operands=[])
  inst4 = RdnaInstruction(opcode="s_cbranch_vccnz", operands=[RdnaLabel(name="L1")])
  inst5 = RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), RdnaImmediate(value=1.5, is_hex=False)])
  inst6 = RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), RdnaImmediate(value=0x10, is_hex=True)])
  inst7 = RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), "some_string"])
  inst8 = RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), "array[0]"])
  label1 = RdnaLabel(name="L1")

  cst_module = synthesizer.to_python([inst1, inst2, inst3, inst4, inst5, inst6, inst7, inst8, label1])
  assert isinstance(cst_module, cst.Module)


def test_rdna_backend():
  """Test element."""
  semantics = MockSemanticsManager()
  backend = RdnaBackend(semantics)

  graph = LogicalGraph()
  n_in = LogicalNode(id="in1", kind="Input")
  graph.nodes.append(n_in)

  result = backend.compile(graph)
  assert "RDNA Code Generation Initialized" in result


def test_rdna_backend_default_semantics():
  """Test element."""
  backend = RdnaBackend()
  assert backend.synthesizer.semantics is not None
