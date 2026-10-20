"""Test suite for rdna synthesizer."""

import typing
from pydantic import JsonValue


import pytest

from ml_switcheroo.core.compiler.backends.rdna.synthesizer import MAX_SGPR, RdnaBackend, RdnaSynthesizer
from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaImmediate,
  RdnaInstruction,
  RdnaLabel,
  RdnaLabelRef,
  RdnaMemory,
  RdnaSGPR,
  RdnaVGPR,
)
from ml_switcheroo.core.compiler.isa.allocator import GenericRegisterAllocator as RegisterAllocator

from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaOperand
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode


class DummyOperand(RdnaOperand):
  """Test helper class DummyOperand."""

  def __init__(self, raw: str) -> None:
    """Initialize test helper instance.

    Args:
        raw: Test argument or fixture.
    """
    self.raw = raw

  def __str__(self) -> str:
    """Execute str helper logic.

    Returns:
        Computed test result or fixture value.
    """
    return self.raw


def test_register_allocator() -> None:
  """Verify register allocator behavior."""
  allocator = RegisterAllocator(256, lambda x: RdnaVGPR(index=x))
  vgpr = allocator.get_register("var1")
  assert vgpr.index == 0
  vgpr2 = allocator.get_register("var1")
  assert vgpr2.index == 0

  sgpr = allocator.get_register("var1")
  assert sgpr.index == 0

  tmp_vgpr = allocator.allocate_temp()
  assert tmp_vgpr.index == 1

  tmp_sgpr = allocator.allocate_temp()
  assert tmp_sgpr.index == 2

  allocator.reset()
  assert allocator.get_register("var3").index == 0


def test_register_allocator_overflow() -> None:
  """Verify register allocator overflow behavior."""
  allocator = RegisterAllocator(256, lambda x: RdnaVGPR(index=x))
  allocator._free_pool = []
  with pytest.raises(ValueError, match="overflow"):
    allocator.allocate_temp()

  allocator._next_sgpr = MAX_SGPR  # type: ignore[attr-defined]
  with pytest.raises(ValueError, match="overflow"):
    allocator.allocate_temp()


def test_synthesizer_from_graph() -> None:
  """Verify synthesizer from graph behavior."""

  class MockSemantics:
    """Test helper class MockSemantics."""

    def get_definition(self, kind: str) -> typing.Optional[tuple[str, dict[str, JsonValue]]]:
      """Execute get definition helper logic.

      Args:
          kind: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      if kind == "UnknownNode":
        return None
      return ("abstract.Linear", {})

    def resolve_variant(self, abstract_id: str, target: str) -> typing.Optional[dict[str, JsonValue]]:
      """Execute resolve variant helper logic.

      Args:
          abstract_id: Test argument or fixture.
          target: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return {"api": "v_nop", "args": []}

  synth = RdnaSynthesizer(semantics=MockSemantics())

  def dummy_expander(
    alloc: RegisterAllocator[RdnaVGPR],
    nid: str,
    meta: typing.Mapping[str, JsonValue],
  ) -> list[RdnaInstruction]:
    """Execute dummy expander helper logic.

    Args:
        alloc: Test argument or fixture.
        nid: Test argument or fixture.
        meta: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return [RdnaInstruction(opcode="s_nop", operands=[])]

  synth.macro_registry = {"Linear": dummy_expander}  # type: ignore[dict-item]

  graph = LogicalGraph(
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

  nodes = synth.from_graph(graph)
  assert len(nodes) > 0


def test_synthesizer_from_graph_exact_macro() -> None:
  """Verify synthesizer from graph exact macro behavior."""

  class MockSemantics:
    """Test helper class MockSemantics."""

    def get_definition(self, kind: str) -> typing.Optional[tuple[str, dict[str, JsonValue]]]:
      """Execute get definition helper logic.

      Args:
          kind: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return ("ExactMacro", {})

    def resolve_variant(self, abstract_id: str, target: str) -> typing.Optional[dict[str, JsonValue]]:
      """Execute resolve variant helper logic.

      Args:
          abstract_id: Test argument or fixture.
          target: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return {"api": "v_add", "args": ["a", "b", "c"]}

  synth = RdnaSynthesizer(semantics=MockSemantics())

  def dummy_expander(
    alloc: RegisterAllocator[RdnaVGPR],
    nid: str,
    meta: typing.Mapping[str, JsonValue],
  ) -> list[RdnaInstruction]:
    """Execute dummy expander helper logic.

    Args:
        alloc: Test argument or fixture.
        nid: Test argument or fixture.
        meta: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return [RdnaInstruction(opcode="s_nop", operands=[])]

  synth.macro_registry = {"ExactMacro": dummy_expander}  # type: ignore[dict-item]
  graph = LogicalGraph(
    nodes={n.id: n for n in [LogicalNode(id="node", op_type="MyOp"), LogicalNode(id="unmatched", op_type="UnmatchedOp")]},
    edges=[],
  )
  nodes = synth.from_graph(graph)
  assert len(nodes) > 0


def test_synthesizer_to_python() -> None:
  """Verify synthesizer to python behavior."""
  synth = RdnaSynthesizer(semantics=None)

  nodes = [
    RdnaInstruction(opcode="v_add_f32", operands=[RdnaVGPR(index=0), RdnaVGPR(index=1), RdnaVGPR(index=2)]),
    RdnaLabel(name="label1"),
    RdnaInstruction(opcode="s_nop", operands=[]),
    RdnaInstruction(opcode="v_mov_b32", operands=[RdnaVGPR(index=0), DummyOperand("s[0:1]")]),
    RdnaInstruction(opcode="s_mov_b32", operands=[RdnaSGPR(index=0), DummyOperand("#hashtag")]),
  ]
  tree = synth.to_python(nodes)
  code = tree.code
  assert "v0 =" in code
  assert "s_nop" in code
  assert "s_0_1" in code
  assert "'#hashtag'" in code
  assert "RdnaLabel" in code


def test_synthesizer_to_python_operands() -> None:
  """Verify synthesizer to python operands behavior."""
  synth = RdnaSynthesizer(semantics=None)
  nodes = [
    RdnaInstruction(opcode="v_mov_b32", operands=[RdnaVGPR(index=0), RdnaImmediate(value=42)]),
    RdnaInstruction(opcode="s_branch", operands=[RdnaLabelRef(name="label1")]),
    RdnaInstruction(opcode="global_store", operands=[RdnaMemory(base=RdnaVGPR(index=0), offset=4)]),
    RdnaInstruction(opcode="v_mov_b32", operands=[RdnaVGPR(index=1), RdnaImmediate(value=1.5, is_hex=False)]),
    RdnaInstruction(opcode="v_mov_b32", operands=[RdnaVGPR(index=2), RdnaImmediate(value=1, is_hex=True)]),
  ]
  tree = synth.to_python(nodes)
  code = tree.code
  assert "42" in code
  assert "1.5" in code
  assert "global_store" in code


def test_backend_compile() -> None:
  """Verify backend compile behavior."""
  backend = RdnaBackend()
  graph = LogicalGraph(nodes={n.id: n for n in [LogicalNode(id="in1", op_type="Input")]}, edges=[])
  code = backend.compile(graph)
  assert "RDNA Code Generation Initialized" in code


def test_synthesizer_from_graph_unmapped() -> None:
  """Verify synthesizer from graph unmapped behavior."""

  class MockSemantics:
    """Test helper class MockSemantics."""

    def get_definition(self, kind: str) -> typing.Optional[tuple[str, dict[str, JsonValue]]]:
      """Execute get definition helper logic.

      Args:
          kind: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return ("abstract", {})

    def resolve_variant(self, abstract_id: str, target: str) -> typing.Optional[dict[str, JsonValue]]:
      """Execute resolve variant helper logic.

      Args:
          abstract_id: Test argument or fixture.
          target: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return None

  synth = RdnaSynthesizer(semantics=MockSemantics())
  synth.macro_registry = {}

  graph = LogicalGraph(
    nodes={
      n.id: n
      for n in [
        LogicalNode(id="op1", op_type="UnknownOp"),
      ]
    },
    edges=[],
  )
  nodes = synth.from_graph(graph)
  assert len(nodes) == 1
  assert "Unmapped Op" in str(nodes[0])


def test_synthesizer_from_graph_with_sources() -> None:
  """Verify synthesizer from graph with sources behavior."""

  class MockSemantics:
    """Test helper class MockSemantics."""

    def get_definition(self, kind: str) -> typing.Optional[tuple[str, dict[str, JsonValue]]]:
      """Execute get definition helper logic.

      Args:
          kind: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return ("abstract", {})

    def resolve_variant(self, abstract_id: str, target: str) -> typing.Optional[dict[str, JsonValue]]:
      """Execute resolve variant helper logic.

      Args:
          abstract_id: Test argument or fixture.
          target: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return {"api": "v_add", "args": ["a"]}

  synth = RdnaSynthesizer(semantics=MockSemantics())
  synth.macro_registry = {}

  graph = LogicalGraph(
    nodes={
      n.id: n
      for n in [
        LogicalNode(id="in1", op_type="Input"),
        LogicalNode(id="op1", op_type="MyOp"),
      ]
    },
    edges=[LogicalEdge("in1", "op1")],
  )
  nodes = synth.from_graph(graph)
  assert len(nodes) > 0


def test_register_allocator_hit_sgpr_cache() -> None:
  """Verify register allocator hit sgpr cache behavior."""
  allocator = RegisterAllocator(256, lambda x: RdnaVGPR(index=x))
  first = allocator.get_register("var1")
  _res = allocator.get_register("var1")
  assert _res == first
  assert _res.index == 0


def test_synthesizer_macro_directives_and_fallbacks() -> None:
  """Test macro directives with semicolons, without semicolons, and invalid opcodes."""

  class MockSemantics:
    """Mock semantics resolving macro directives."""

    def get_definition(self, kind: str) -> typing.Optional[tuple[str, dict[str, JsonValue]]]:
      """Lookup definition.

      Args:
          kind: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return (kind, {})

    def resolve_variant(self, abstract_id: str, target: str) -> typing.Optional[dict[str, JsonValue]]:
      """Resolve variant.

      Args:
          abstract_id: Test argument or fixture.
          target: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      if abstract_id == "op_macro_semicolon":
        return {"api": "; Macro.LayerNorm"}
      if abstract_id == "op_macro_unknown":
        return {"api": "Macro.CustomUnknown"}
      if abstract_id == "op_semicolon_comment":
        return {"api": "; unmapped custom comment"}
      return None

  synth = RdnaSynthesizer(semantics=MockSemantics())
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
  """Test synthesizer initialization when macros.json is missing or contains missing functions.

  Args:
      monkeypatch: Test argument or fixture.
  """
  import json
  import os
  import unittest.mock as mock

  import ml_switcheroo.core.compiler.backends.rdna.synthesizer as synth_mod

  # Test when macros.json does not exist
  monkeypatch.setattr(os.path, "exists", lambda p: False)
  synth = synth_mod.RdnaSynthesizer(semantics=None)
  assert synth.macro_registry == {}

  # Test when macros.json has a function not in rdna_macros
  monkeypatch.setattr(os.path, "exists", lambda p: True)
  m_open = mock.mock_open(read_data=json.dumps({"dummy_op": "non_existent_func"}))
  monkeypatch.setattr("builtins.open", m_open)
  synth2 = synth_mod.RdnaSynthesizer(semantics=None)
  assert "dummy_op" not in synth2.macro_registry


def test_synthesizer_from_graph_edge_cases() -> None:
  """Test edge cases in RdnaSynthesizer.from_graph."""

  class MockSemantics:
    """Mock semantics."""

    def get_definition(self, kind: str) -> typing.Optional[tuple[str, dict[str, JsonValue]]]:
      """Lookup definition.

      Args:
          kind: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      if kind == "nn.Dotted":
        return ("DottedOp", {})
      return None

    def resolve_variant(self, abstract_id: str, target: str) -> typing.Optional[dict[str, JsonValue]]:
      """Resolve variant.

      Args:
          abstract_id: Test argument or fixture.
          target: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      if abstract_id == "DottedOp":
        return {"api": "v_nop"}
      return None

  synth = RdnaSynthesizer(semantics=MockSemantics())
  graph = LogicalGraph(
    nodes={
      n.id: n
      for n in [
        LogicalNode(id="empty", op_type=""),
        LogicalNode(id="dot_node", op_type="torch.nn.Dotted"),
        LogicalNode(id="out_empty", op_type="Output"),
      ]
    }
  )
  nodes = synth.from_graph(graph)
  assert len(nodes) >= 2


def test_synthesizer_to_python_comment() -> None:
  """Test to_python with RdnaComment node."""
  synth = RdnaSynthesizer(semantics=None)
  nodes = [
    RdnaComment(text="a comment"),
  ]
  tree = synth.to_python(nodes)
  assert tree.body == []


def test_backend_compile_with_semantics() -> None:
  """Test RdnaBackend initialization with explicit semantics."""

  class MockSemantics:
    """Mock semantics."""

    def get_definition(self, kind: str) -> typing.Optional[tuple[str, dict[str, JsonValue]]]:
      """Return None.

      Args:
          kind: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return None

    def resolve_variant(self, abstract_id: str, target: str) -> typing.Optional[dict[str, JsonValue]]:
      """Return None.

      Args:
          abstract_id: Test argument or fixture.
          target: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return None

  backend = RdnaBackend(semantics=MockSemantics())  # type: ignore[arg-type]
  graph = LogicalGraph(nodes={n.id: n for n in [LogicalNode(id="in1", op_type="Input")]}, edges=[])
  code = backend.compile(graph)
  assert "RDNA Code Generation Initialized" in code
