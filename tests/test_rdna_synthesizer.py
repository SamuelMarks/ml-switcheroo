"""Docstring."""

from typing import Any, Dict, List, Optional, Tuple

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
  RdnaLabelRef,
  RdnaMemory,
  RdnaNode,
  RdnaOperand,
  RdnaSGPR,
  RdnaVGPR,
)
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode


class DummyOperand(RdnaOperand):
  """Docstring."""

  def __init__(self, raw: str) -> None:
    """Docstring."""
    self.raw = raw

  def __str__(self) -> str:
    """Docstring."""
    return self.raw


def test_register_allocator() -> None:
  """Docstring."""
  allocator: RegisterAllocator = RegisterAllocator()
  vgpr: RdnaVGPR = allocator.get_vector_register("var1")
  assert vgpr.index == 0
  vgpr2: RdnaVGPR = allocator.get_vector_register("var1")
  assert vgpr2.index == 0

  sgpr: RdnaSGPR = allocator.get_scalar_register("var1")
  assert sgpr.index == 0

  tmp_vgpr: RdnaVGPR = allocator.allocate_vector_temp()
  assert tmp_vgpr.index == 1

  tmp_sgpr: RdnaSGPR = allocator.allocate_scalar_temp()
  assert tmp_sgpr.index == 1

  allocator.reset()
  assert allocator.get_vector_register("var3").index == 0


def test_register_allocator_overflow() -> None:
  """Docstring."""
  allocator: RegisterAllocator = RegisterAllocator()
  allocator._next_vgpr = MAX_VGPR
  with pytest.raises(ValueError, match="RdnaVGPR overflow"):
    allocator.allocate_vector_temp()

  allocator._next_sgpr = MAX_SGPR
  with pytest.raises(ValueError, match="RdnaSGPR overflow"):
    allocator.allocate_scalar_temp()


def test_synthesizer_from_graph() -> None:
  """Docstring."""

  class MockSemantics:
    def get_definition(self, kind: str) -> Optional[Tuple[str, dict]]:
      if kind == "UnknownNode":
        return None
      return ("abstract.Linear", {})

    def resolve_variant(self, abstract_id: str, target: str) -> Optional[Dict[str, Any]]:
      return {"api": "v_nop", "args": []}

  synth: RdnaSynthesizer = RdnaSynthesizer(semantics=MockSemantics())

  def dummy_expander(alloc: RegisterAllocator, nid: str, meta: dict) -> List[RdnaNode]:
    return [RdnaInstruction("s_nop", [])]

  synth.macro_registry = {"Linear": dummy_expander}

  graph: LogicalGraph = LogicalGraph(
    nodes=[
      LogicalNode(id="in1", kind="Input"),
      LogicalNode(id="conv1", kind="Conv2d", metadata={"k": 3}),
      LogicalNode(id="lin1", kind="Linear"),
      LogicalNode(id="out1", kind="Output"),
      LogicalNode(id="unkn1", kind="UnknownNode"),
    ],
    edges=[LogicalEdge("in1", "conv1"), LogicalEdge("conv1", "out1"), LogicalEdge("in1", "out1")],
  )

  nodes: List[RdnaNode] = synth.from_graph(graph)
  assert len(nodes) > 0


def test_synthesizer_from_graph_exact_macro() -> None:
  """Docstring."""

  class MockSemantics:
    def get_definition(self, kind: str) -> Optional[Tuple[str, dict]]:
      return ("ExactMacro", {})

    def resolve_variant(self, abstract_id: str, target: str) -> Optional[Dict[str, Any]]:
      return {"api": "v_add", "args": ["a", "b", "c"]}

  synth: RdnaSynthesizer = RdnaSynthesizer(semantics=MockSemantics())

  def dummy_expander(alloc: RegisterAllocator, nid: str, meta: dict) -> List[RdnaNode]:
    return [RdnaInstruction("s_nop", [])]

  synth.macro_registry = {"ExactMacro": dummy_expander}
  graph: LogicalGraph = LogicalGraph(
    nodes=[LogicalNode(id="node", kind="MyOp"), LogicalNode(id="unmatched", kind="UnmatchedOp")], edges=[]
  )
  nodes: List[RdnaNode] = synth.from_graph(graph)
  assert len(nodes) > 0


def test_synthesizer_to_python() -> None:
  """Docstring."""
  synth: RdnaSynthesizer = RdnaSynthesizer(semantics=None)

  nodes: List[RdnaNode] = [
    RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), RdnaVGPR(index=1), RdnaVGPR(index=2)]),
    RdnaLabel(name="label1"),
    RdnaInstruction(opcode="s_nop", operands=[]),
    RdnaInstruction(opcode="v_mov_b32", operands=[RdnaVGPR(index=0), DummyOperand("s[0:1]")]),
    RdnaInstruction(opcode="s_mov_b32", operands=[RdnaSGPR(index=0), DummyOperand("#hashtag")]),
  ]
  tree: cst.Module = synth.to_python(nodes)
  code: str = tree.code
  assert "v0 =" in code
  assert "s_nop" in code
  assert "s_0_1" in code
  assert "'#hashtag'" in code
  assert "RdnaLabel" in code


def test_synthesizer_to_python_operands() -> None:
  """Docstring."""
  synth: RdnaSynthesizer = RdnaSynthesizer(semantics=None)
  nodes: List[RdnaNode] = [
    RdnaInstruction(opcode="v_mov_b32", operands=[RdnaVGPR(index=0), RdnaImmediate(value=42)]),
    RdnaInstruction(opcode="s_branch", operands=[RdnaLabelRef(name="label1")]),
    RdnaInstruction(opcode="global_store", operands=[RdnaMemory(base=RdnaVGPR(index=0), offset=4)]),
    RdnaInstruction(opcode="v_mov_b32", operands=[RdnaVGPR(index=1), RdnaImmediate(value=1.5, is_hex=False)]),
    RdnaInstruction(opcode="v_mov_b32", operands=[RdnaVGPR(index=2), RdnaImmediate(value=1, is_hex=True)]),
  ]
  tree: cst.Module = synth.to_python(nodes)
  code: str = tree.code
  assert "42" in code
  assert "1.5" in code
  assert "global_store" in code


def test_backend_compile() -> None:
  """Docstring."""
  backend: RdnaBackend = RdnaBackend()
  graph: LogicalGraph = LogicalGraph(nodes=[LogicalNode(id="in1", kind="Input")], edges=[])
  code: str = backend.compile(graph)
  assert "RDNA Code Generation Initialized" in code


def test_synthesizer_from_graph_unmapped() -> None:
  """Docstring."""

  class MockSemantics:
    def get_definition(self, kind: str) -> Optional[Tuple[str, dict]]:
      return ("abstract", {})

    def resolve_variant(self, abstract_id: str, target: str) -> Optional[Dict[str, Any]]:
      return None

  synth: RdnaSynthesizer = RdnaSynthesizer(semantics=MockSemantics())
  synth.macro_registry = {}

  graph: LogicalGraph = LogicalGraph(
    nodes=[
      LogicalNode(id="op1", kind="UnknownOp"),
    ],
    edges=[],
  )
  nodes: List[RdnaNode] = synth.from_graph(graph)
  assert len(nodes) == 1
  assert "Unmapped Op" in str(nodes[0])


def test_synthesizer_from_graph_with_sources() -> None:
  """Docstring."""

  class MockSemantics:
    def get_definition(self, kind: str) -> Optional[Tuple[str, dict]]:
      return ("abstract", {})

    def resolve_variant(self, abstract_id: str, target: str) -> Optional[Dict[str, Any]]:
      return {"api": "v_add", "args": ["a"]}

  synth: RdnaSynthesizer = RdnaSynthesizer(semantics=MockSemantics())
  synth.macro_registry = {}

  graph: LogicalGraph = LogicalGraph(
    nodes=[
      LogicalNode(id="in1", kind="Input"),
      LogicalNode(id="op1", kind="MyOp"),
    ],
    edges=[LogicalEdge("in1", "op1")],
  )
  nodes: List[RdnaNode] = synth.from_graph(graph)
  assert len(nodes) > 0


def test_register_allocator_hit_sgpr_cache() -> None:
  """Docstring."""
  allocator: RegisterAllocator = RegisterAllocator()
  allocator.get_scalar_register("var1")
  allocator.get_scalar_register("var1")
