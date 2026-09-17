"""Docstring."""

from typing import Any, Dict, List, Optional, Tuple, cast

import libcst as cst
import pytest

from ml_switcheroo.core.compiler.backends.nvidia_sass.backend import NvidiaSassBackend
from ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer import RegisterAllocator, NvidiaSassSynthesizer
from ml_switcheroo.core.compiler.frontends.nvidia_sass.cst import (
  NvidiaSassComment,
  NvidiaSassImmediate,
  NvidiaSassInstruction,
  NvidiaSassLabel,
  NvidiaSassMemory,
  NvidiaSassNode,
  NvidiaSassOperand,
  NvidiaSassPredicate,
  NvidiaSassRegister,
)
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode


class DummyNvidiaSassOperand(NvidiaSassOperand):
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
  reg: NvidiaSassRegister = allocator.get_register("var1")
  assert reg.name == "R0"
  reg2: NvidiaSassRegister = allocator.get_register("var1")
  assert reg2.name == "R0"

  tmp_reg: NvidiaSassRegister = allocator.allocate_temp()
  assert tmp_reg.name == "R1"

  allocator.reset()
  assert allocator.get_register("var3").name == "R0"


def test_register_allocator_overflow() -> None:
  """Docstring."""
  allocator: RegisterAllocator = RegisterAllocator()
  allocator._free_pool = []
  with pytest.raises(ValueError, match="NvidiaSassRegister overflow"):
    allocator.allocate_temp()


def test_synthesizer_from_graph() -> None:
  """Docstring."""

  class MockSemantics:
    """Docstring."""

    def get_definition(self, kind: str) -> Optional[Tuple[str, Dict[str, Any]]]:
      """Docstring."""
      if kind == "UnknownNode":
        return None
      if kind == "DirectMatch":
        return ("DirectMatch", {})
      if kind == "FallbackOp":
        return ("FallbackOp", {})
      return ("abstract.Linear", {})

    def resolve_variant(self, abstract_id: str, target: str) -> Optional[Dict[str, Any]]:
      """Docstring."""
      if abstract_id == "FallbackOp":
        return {"api": "FADD", "args": []}
      return {"api": "NOP", "args": []}

  synth: NvidiaSassSynthesizer = NvidiaSassSynthesizer(semantics=cast(Any, MockSemantics()))

  def dummy_expander(alloc: RegisterAllocator, nid: str, meta: Dict[str, Any]) -> List[NvidiaSassNode]:
    """Docstring."""
    return [NvidiaSassInstruction(opcode="NOP", operands=[])]

  synth.macro_registry = {"Linear": dummy_expander, "DirectMatch": dummy_expander}

  graph: LogicalGraph = LogicalGraph(
    nodes={
      n.id: n
      for n in [
        LogicalNode(id="in1", op_type="Input"),
        LogicalNode(id="conv1", op_type="Conv2d", attributes={"k": "3"}),
        LogicalNode(id="lin1", op_type="Linear"),
        LogicalNode(id="dir1", op_type="DirectMatch"),
        LogicalNode(id="fall1", op_type="FallbackOp"),
        LogicalNode(id="out1", op_type="Output"),
        LogicalNode(id="unkn1", op_type="UnknownNode"),
      ]
    },
    edges=[
      LogicalEdge("in1", "conv1"),
      LogicalEdge("conv1", "lin1"),
      LogicalEdge("in1", "dir1"),
      LogicalEdge("in1", "fall1"),
      LogicalEdge("lin1", "out1"),
    ],
  )

  nodes: List[NvidiaSassNode] = synth.from_graph(graph)
  assert len(nodes) > 0


def test_synthesizer_from_graph_unmapped() -> None:
  """Docstring."""

  class MockSemantics:
    """Docstring."""

    def get_definition(self, kind: str) -> Optional[Tuple[str, Dict[str, Any]]]:
      """Docstring."""
      return ("abstract", {})

    def resolve_variant(self, abstract_id: str, target: str) -> Optional[Dict[str, Any]]:
      """Docstring."""
      return None

  synth: NvidiaSassSynthesizer = NvidiaSassSynthesizer(semantics=cast(Any, MockSemantics()))
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
  nodes: List[NvidiaSassNode] = synth.from_graph(graph)
  assert len(nodes) == 1
  assert "Unmapped Op" in str(nodes[0])


def test_synthesizer_to_python() -> None:
  """Docstring."""
  synth: NvidiaSassSynthesizer = NvidiaSassSynthesizer(semantics=cast(Any, None))

  nodes: List[NvidiaSassNode] = [
    NvidiaSassComment(text="BEGIN Conv2d"),
    NvidiaSassInstruction(
      opcode="FADD",
      operands=[NvidiaSassRegister(name="R0"), NvidiaSassRegister(name="R1"), NvidiaSassRegister(name="R2")],
    ),
    NvidiaSassLabel(name="label1"),
    NvidiaSassInstruction(opcode="NOP", operands=[]),
    NvidiaSassInstruction(
      opcode="MOV",
      operands=[NvidiaSassRegister(name="R0"), DummyNvidiaSassOperand("R[0:1]")],
      predicate=NvidiaSassPredicate(name="P0"),
    ),
    NvidiaSassInstruction(opcode="MOV", operands=[NvidiaSassRegister(name="R0"), DummyNvidiaSassOperand("#hashtag")]),
    NvidiaSassInstruction(
      opcode="MOV", operands=[DummyNvidiaSassOperand("-R0"), NvidiaSassImmediate(value=42)]
    ),  # dest is not identifier
  ]
  tree: cst.Module = synth.to_python(nodes)
  code: str = getattr(tree, "code")
  assert "R0 =" in code
  assert "NOP" in code
  assert "'R[0:1]'" in code
  assert "'#hashtag'" in code
  assert "NvidiaSassLabel" in code
  assert "predicate" in code


def test_synthesizer_to_python_operands() -> None:
  """Docstring."""
  synth: NvidiaSassSynthesizer = NvidiaSassSynthesizer(semantics=cast(Any, None))
  nodes: List[NvidiaSassNode] = [
    NvidiaSassInstruction(opcode="MOV", operands=[NvidiaSassRegister(name="R0"), NvidiaSassImmediate(value=42)]),
    NvidiaSassInstruction(opcode="BRA", operands=[NvidiaSassLabel(name="label1")]),
    NvidiaSassInstruction(opcode="STG", operands=[NvidiaSassMemory(base=NvidiaSassRegister(name="R0"), offset=4)]),
    NvidiaSassInstruction(
      opcode="MOV", operands=[NvidiaSassRegister(name="R1"), NvidiaSassImmediate(value=1.5, is_hex=False)]
    ),
    NvidiaSassInstruction(
      opcode="MOV", operands=[NvidiaSassRegister(name="R2"), NvidiaSassImmediate(value=1, is_hex=True)]
    ),
  ]
  tree: cst.Module = synth.to_python(nodes)
  code: str = getattr(tree, "code")
  assert "42" in code
  assert "1.5" in code
  assert "STG" in code


def test_backend_compile() -> None:
  """Docstring."""
  backend: NvidiaSassBackend = NvidiaSassBackend()
  graph: LogicalGraph = LogicalGraph(nodes={n.id: n for n in [LogicalNode(id="in1", op_type="Input")]}, edges=[])
  code: str = backend.compile(graph)
  assert "Input in1" in code


def test_synthesizer_from_graph_prefixed_kind() -> None:
  """Test resolving node kinds with namespace prefixes (e.g. rdna.v_add_f32)."""

  class MockSemantics:
    """Mock semantics supporting suffix lookup."""

    def get_definition(self, kind: str) -> Optional[Tuple[str, Dict[str, Any]]]:
      """Resolve definition only for non-prefixed suffix."""
      if kind == "v_add_f32":
        return ("add", {})
      return None

    def resolve_variant(self, abstract_id: str, target: str) -> Optional[Dict[str, Any]]:
      """Resolve variant for abstract operation."""
      if abstract_id == "add":
        return {"api": "FADD"}
      return None

  synth: NvidiaSassSynthesizer = NvidiaSassSynthesizer(semantics=cast(Any, MockSemantics()))
  graph: LogicalGraph = LogicalGraph(
    nodes={
      n.id: n
      for n in [
        LogicalNode(id="n1", op_type="rdna.v_add_f32"),
      ]
    },
    edges=[],
  )
  nodes = synth.from_graph(graph)
  assert len(nodes) == 1
  assert getattr(nodes[0], "opcode", "") == "FADD"


def test_register_allocator_record_usage_untracked() -> None:
  """Test record_usage with untracked variable."""
  allocator: RegisterAllocator = RegisterAllocator()
  allocator.record_usage("untracked_var")


def test_synthesizer_init_macros_missing_and_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test synthesizer initialization when macros.json is missing or contains missing functions."""
  import json
  import os
  import unittest.mock as mock
  import ml_switcheroo.core.compiler.backends.nvidia_sass.synthesizer as synth_mod

  # Test when macros.json does not exist
  monkeypatch.setattr(os.path, "exists", lambda p: False)
  synth: NvidiaSassSynthesizer = synth_mod.NvidiaSassSynthesizer(semantics=cast(Any, None))
  assert synth.macro_registry == {}

  # Test when macros.json has a function not in sass_macros
  monkeypatch.setattr(os.path, "exists", lambda p: True)
  m_open = mock.mock_open(read_data=json.dumps({"dummy_op": "non_existent_func"}))
  monkeypatch.setattr("builtins.open", m_open)
  synth2: NvidiaSassSynthesizer = synth_mod.NvidiaSassSynthesizer(semantics=cast(Any, None))
  assert "dummy_op" not in synth2.macro_registry


def test_synthesizer_from_graph_edges_and_empty_output() -> None:
  """Test edge cases in from_graph: multi-input node, disconnected output, and empty abstract_id."""

  class MockSemantics:
    """Mock semantics for edge cases."""

    def get_definition(self, kind: str) -> Optional[Tuple[str, Dict[str, Any]]]:
      """Return definition."""
      if kind == "":
        return None
      return (kind, {})

    def resolve_variant(self, abstract_id: str, target: str) -> Optional[Dict[str, Any]]:
      """Return variant."""
      if abstract_id == "add":
        return {"api": "FADD"}
      return None

  synth: NvidiaSassSynthesizer = NvidiaSassSynthesizer(semantics=cast(Any, MockSemantics()))
  graph: LogicalGraph = LogicalGraph(
    nodes={
      n.id: n
      for n in [
        LogicalNode(id="in1", op_type="Input"),
        LogicalNode(id="in2", op_type="Input"),
        LogicalNode(id="add1", op_type="add"),
        LogicalNode(id="empty_kind", op_type=""),
        LogicalNode(id="disconnected_out", op_type="Output"),
      ]
    },
    edges=[
      LogicalEdge("in1", "add1"),
      LogicalEdge("in2", "add1"),
    ],
  )
  nodes: List[NvidiaSassNode] = synth.from_graph(graph)
  assert any("Unmapped Op:  (empty_kind)" in str(n) for n in nodes)


class OtherNode(NvidiaSassNode):
  """Other node type for testing to_python branch."""

  def __str__(self) -> str:
    """Return string representation."""
    return "; other"


def test_synthesizer_to_python_more_branches() -> None:
  """Test to_python with non-directive comment, unknown node, and non-register valid identifier target."""
  synth: NvidiaSassSynthesizer = NvidiaSassSynthesizer(semantics=cast(Any, None))
  nodes: List[NvidiaSassNode] = [
    NvidiaSassComment(text="Arbitrary comment without begin or end"),
    OtherNode(),
    NvidiaSassInstruction(
      opcode="MOV",
      operands=[DummyNvidiaSassOperand("valid_ident"), NvidiaSassImmediate(value=10)],
    ),
  ]
  tree: cst.Module = synth.to_python(nodes)
  code: str = getattr(tree, "code")
  assert "valid_ident = " in code
