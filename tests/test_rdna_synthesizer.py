"""Docstring."""

from typing import Any, Dict, List, Optional, Tuple, cast

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
    """Docstring."""

    def get_definition(self, kind: str) -> Optional[Tuple[str, dict]]:
      """Docstring."""
      if kind == "UnknownNode":
        return None
      return ("abstract.Linear", {})

    def resolve_variant(self, abstract_id: str, target: str) -> Optional[Dict[str, Any]]:
      """Docstring."""
      return {"api": "v_nop", "args": []}

  synth: RdnaSynthesizer = RdnaSynthesizer(semantics=cast(Any, MockSemantics()))

  def dummy_expander(alloc: RegisterAllocator, nid: str, meta: dict) -> List[RdnaNode]:
    """Docstring."""
    return [RdnaInstruction(opcode="s_nop", operands=[])]

  synth.macro_registry = {"Linear": dummy_expander}

  graph: LogicalGraph = LogicalGraph(
    nodes={
      n.id: n
      for n in [
        LogicalNode(id="in1", op_type="Input"),
        LogicalNode(id="conv1", op_type="Conv2d", attributes={"k": "3"}),
        LogicalNode(id="lin1", op_type="Linear"),
        LogicalNode(id="out1", op_type="Output"),
        LogicalNode(id="unkn1", op_type="UnknownNode"),
      ]
    },
    edges=[LogicalEdge("in1", "conv1"), LogicalEdge("conv1", "out1"), LogicalEdge("in1", "out1")],
  )

  nodes: List[RdnaNode] = synth.from_graph(graph)
  assert len(nodes) > 0


def test_synthesizer_from_graph_exact_macro() -> None:
  """Docstring."""

  class MockSemantics:
    """Docstring."""

    def get_definition(self, kind: str) -> Optional[Tuple[str, dict]]:
      """Docstring."""
      return ("ExactMacro", {})

    def resolve_variant(self, abstract_id: str, target: str) -> Optional[Dict[str, Any]]:
      """Docstring."""
      return {"api": "v_add", "args": ["a", "b", "c"]}

  synth: RdnaSynthesizer = RdnaSynthesizer(semantics=cast(Any, MockSemantics()))

  def dummy_expander(alloc: RegisterAllocator, nid: str, meta: dict) -> List[RdnaNode]:
    """Docstring."""
    return [RdnaInstruction(opcode="s_nop", operands=[])]

  synth.macro_registry = {"ExactMacro": dummy_expander}
  graph: LogicalGraph = LogicalGraph(
    nodes={n.id: n for n in [LogicalNode(id="node", op_type="MyOp"), LogicalNode(id="unmatched", op_type="UnmatchedOp")]},
    edges=[],
  )
  nodes: List[RdnaNode] = synth.from_graph(graph)
  assert len(nodes) > 0


def test_synthesizer_to_python() -> None:
  """Docstring."""
  synth: RdnaSynthesizer = RdnaSynthesizer(semantics=cast(Any, None))

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
  synth: RdnaSynthesizer = RdnaSynthesizer(semantics=cast(Any, None))
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
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in [LogicalNode(id="in1", op_type="Input")]}, edges=[])
  code: str = backend.compile(graph)
  assert "RDNA Code Generation Initialized" in code


def test_synthesizer_from_graph_unmapped() -> None:
  """Docstring."""

  class MockSemantics:
    """Docstring."""

    def get_definition(self, kind: str) -> Optional[Tuple[str, dict]]:
      """Docstring."""
      return ("abstract", {})

    def resolve_variant(self, abstract_id: str, target: str) -> Optional[Dict[str, Any]]:
      """Docstring."""
      return None

  synth: RdnaSynthesizer = RdnaSynthesizer(semantics=cast(Any, MockSemantics()))
  synth.macro_registry = {}

  graph: LogicalGraph = LogicalGraph(
    nodes={
      n.id: n
      for n in [
        LogicalNode(id="op1", op_type="UnknownOp"),
      ]
    },
    edges=[],
  )
  nodes: List[RdnaNode] = synth.from_graph(graph)
  assert len(nodes) == 1
  assert "Unmapped Op" in str(nodes[0])


def test_synthesizer_from_graph_with_sources() -> None:
  """Docstring."""

  class MockSemantics:
    """Docstring."""

    def get_definition(self, kind: str) -> Optional[Tuple[str, dict]]:
      """Docstring."""
      return ("abstract", {})

    def resolve_variant(self, abstract_id: str, target: str) -> Optional[Dict[str, Any]]:
      """Docstring."""
      return {"api": "v_add", "args": ["a"]}

  synth: RdnaSynthesizer = RdnaSynthesizer(semantics=cast(Any, MockSemantics()))
  synth.macro_registry = {}

  graph: LogicalGraph = LogicalGraph(
    nodes={
      n.id: n
      for n in [
        LogicalNode(id="in1", op_type="Input"),
        LogicalNode(id="op1", op_type="MyOp"),
      ]
    },
    edges=[LogicalEdge("in1", "op1")],
  )
  nodes: List[RdnaNode] = synth.from_graph(graph)
  assert len(nodes) > 0


def test_register_allocator_hit_sgpr_cache() -> None:
  """Docstring."""
  allocator: RegisterAllocator = RegisterAllocator()
  allocator.get_scalar_register("var1")
  allocator.get_scalar_register("var1")


def test_synthesizer_macro_directives_and_fallbacks() -> None:
  """Test macro directives with semicolons, without semicolons, and invalid opcodes."""

  class MockSemantics:
    """Mock semantics resolving macro directives."""

    def get_definition(self, kind: str) -> Optional[Tuple[str, dict]]:
      """Lookup definition."""
      return (kind, {})

    def resolve_variant(self, abstract_id: str, target: str) -> Optional[Dict[str, Any]]:
      """Resolve variant."""
      if abstract_id == "op_macro_semicolon":
        return {"api": "; Macro.LayerNorm"}
      if abstract_id == "op_macro_unknown":
        return {"api": "Macro.CustomUnknown"}
      if abstract_id == "op_semicolon_comment":
        return {"api": "; unmapped custom comment"}
      return None

  synth: RdnaSynthesizer = RdnaSynthesizer(semantics=cast(Any, MockSemantics()))
  graph = LogicalGraph(
    nodes={
      n.id: n
      for n in [
        LogicalNode(id="n1", op_type="op_macro_semicolon"),
        LogicalNode(id="n2", op_type="op_macro_unknown"),
        LogicalNode(id="n3", op_type="op_semicolon_comment"),
      ]
    }
  )

  nodes = synth.from_graph(graph)
  assert len(nodes) > 0
  texts = [str(n) for n in nodes]
  assert any("LayerNorm" in t for t in texts)
  assert any("CustomUnknown" in t for t in texts)
  assert any("Unmapped Op: ; unmapped custom comment" in t for t in texts)


def test_synthesizer_init_macros_missing_and_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test synthesizer initialization when macros.json is missing or contains missing functions."""
  import json
  import os
  import unittest.mock as mock
  import ml_switcheroo.core.compiler.backends.rdna.synthesizer as synth_mod

  # Test when macros.json does not exist
  monkeypatch.setattr(os.path, "exists", lambda p: False)
  synth: synth_mod.RdnaSynthesizer = synth_mod.RdnaSynthesizer(semantics=cast(Any, None))
  assert synth.macro_registry == {}

  # Test when macros.json has a function not in rdna_macros
  monkeypatch.setattr(os.path, "exists", lambda p: True)
  m_open = mock.mock_open(read_data=json.dumps({"dummy_op": "non_existent_func"}))
  monkeypatch.setattr("builtins.open", m_open)
  synth2: synth_mod.RdnaSynthesizer = synth_mod.RdnaSynthesizer(semantics=cast(Any, None))
  assert "dummy_op" not in synth2.macro_registry


def test_synthesizer_from_graph_edge_cases() -> None:
  """Test edge cases in RdnaSynthesizer.from_graph."""

  class MockSemantics:
    """Mock semantics."""

    def get_definition(self, kind: str) -> Optional[Tuple[str, dict]]:
      """Lookup definition."""
      if kind == "nn.Dotted":
        return ("DottedOp", {})
      return None

    def resolve_variant(self, abstract_id: str, target: str) -> Optional[Dict[str, Any]]:
      """Resolve variant."""
      if abstract_id == "DottedOp":
        return {"api": "v_nop"}
      return None

  synth: RdnaSynthesizer = RdnaSynthesizer(semantics=cast(Any, MockSemantics()))
  graph: LogicalGraph = LogicalGraph(
    nodes={
      n.id: n
      for n in [
        LogicalNode(id="empty", op_type=""),
        LogicalNode(id="dot_node", op_type="torch.nn.Dotted"),
        LogicalNode(id="out_empty", op_type="Output"),
      ]
    }
  )
  nodes: List[RdnaNode] = synth.from_graph(graph)
  assert len(nodes) >= 2


def test_synthesizer_to_python_comment() -> None:
  """Test to_python with RdnaComment node."""
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment

  synth: RdnaSynthesizer = RdnaSynthesizer(semantics=cast(Any, None))
  nodes: List[RdnaNode] = [
    RdnaComment(text="a comment"),
  ]
  tree: cst.Module = synth.to_python(nodes)
  assert tree.body == []


def test_backend_compile_with_semantics() -> None:
  """Test RdnaBackend initialization with explicit semantics."""

  class MockSemantics:
    """Mock semantics."""

    def get_definition(self, kind: str) -> Optional[Tuple[str, dict]]:
      """Return None."""
      return None

    def resolve_variant(self, abstract_id: str, target: str) -> Optional[Dict[str, Any]]:
      """Return None."""
      return None

  backend: RdnaBackend = RdnaBackend(semantics=cast(Any, MockSemantics()))
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in [LogicalNode(id="in1", op_type="Input")]}, edges=[])
  code: str = backend.compile(graph)
  assert "RDNA Code Generation Initialized" in code
