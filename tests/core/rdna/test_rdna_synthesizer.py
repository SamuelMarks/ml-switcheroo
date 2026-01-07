"""
Tests for RDNA Synthesizer and Register Allocator.

Verifies:
1.  **Register Allocation**: Dual pools (scalar/vector), overflow handling.
2.  **Target Synthesis**: Graph -> RDNA logic (1:1 mapping).
3.  **Source Synthesis**: RDNA -> Python AST logic.
"""

import pytest
from unittest.mock import MagicMock
import libcst as cst

from ml_switcheroo.core.rdna.synthesizer import (
  RegisterAllocator,
  RdnaSynthesizer,
  MAX_VGPR,
  MAX_SGPR,
)
from ml_switcheroo.core.graph import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.core.rdna.nodes import Instruction, VGPR, SGPR, Comment, Label
from ml_switcheroo.semantics.manager import SemanticsManager

# --- 1. Allocator Tests ---


def test_allocator_dual_pools() -> None:
  """Verify distinct allocation for Scalar and Vector pools."""
  alloc = RegisterAllocator()
  v0 = alloc.get_vector_register("x")
  s0 = alloc.get_scalar_register("cnt")

  assert isinstance(v0, VGPR)
  assert v0.index == 0

  assert isinstance(s0, SGPR)
  assert s0.index == 0


def test_allocator_reuse() -> None:
  """Verify idempotency for same variable name."""
  alloc = RegisterAllocator()
  v_a = alloc.get_vector_register("a")
  v_b = alloc.get_vector_register("a")
  assert v_a.index == v_b.index == 0


def test_allocator_overflow_vgpr() -> None:
  """Verify VGPR limit check."""
  alloc = RegisterAllocator()
  alloc._next_vgpr = MAX_VGPR + 1
  with pytest.raises(ValueError, match="VGPR overflow"):
    alloc.get_vector_register("fail")


def test_allocator_overflow_sgpr() -> None:
  """Verify SGPR limit check."""
  alloc = RegisterAllocator()
  alloc._next_sgpr = MAX_SGPR + 1
  with pytest.raises(ValueError, match="SGPR overflow"):
    alloc.get_scalar_register("fail")


def test_allocator_temps() -> None:
  """Verify automatic temporary allocation."""
  alloc = RegisterAllocator()
  t1 = alloc.allocate_vector_temp()
  t2 = alloc.allocate_scalar_temp()

  assert t1.index == 0
  assert t2.index == 0
  # Next should increment
  t3 = alloc.allocate_vector_temp()
  assert t3.index == 1


# --- 2. Target Synthesis (Graph -> RDNA) ---


@pytest.fixture
def mock_semantics() -> MagicMock:
  mgr = MagicMock(spec=SemanticsManager)

  def get_def(kind):
    if kind == "Add":
      return ("Add", {})
    return None

  def resolve(aid, fw):
    if fw == "rdna" and aid == "Add":
      return {"api": "v_add_f32"}
    return None

  mgr.get_definition.side_effect = get_def
  mgr.resolve_variant.side_effect = resolve
  return mgr


def test_graph_to_rdna_basic_math(mock_semantics: MagicMock) -> None:
  """
  Scenario: x -> Add(y) -> z.
  Expectation:
      ; Input x -> v0
      ; Input y -> v1
      v_add_f32 v2, v0, v1
  """
  synth = RdnaSynthesizer(mock_semantics)
  g = LogicalGraph()
  g.nodes = [
    LogicalNode("x", "Input", {}),
    LogicalNode("y", "Input", {}),
    LogicalNode("z", "Add", {}),
  ]
  g.edges = [LogicalEdge("x", "z"), LogicalEdge("y", "z")]

  nodes = synth.from_graph(g)

  assert len(nodes) == 3
  # Check Inputs
  assert isinstance(nodes[0], Comment)
  assert "** Input x -> v0" in str(nodes[0]).replace(";", "**")

  inst = nodes[2]
  assert isinstance(inst, Instruction)
  assert inst.opcode == "v_add_f32"
  # DST(v2), SRC(v0), SRC(v1)
  assert str(inst.operands[0]) == "v2"
  assert str(inst.operands[1]) == "v0"
  assert str(inst.operands[2]) == "v1"


def test_graph_to_rdna_unmapped(mock_semantics: MagicMock) -> None:
  """Verify fallback comment for unmapped ops."""
  synth = RdnaSynthesizer(mock_semantics)
  g = LogicalGraph()
  g.nodes = [LogicalNode("n1", "MysteryOp", {})]

  nodes = synth.from_graph(g)
  assert len(nodes) == 1
  assert "Unmapped Op: MysteryOp" in str(nodes[0])


# --- 3. Source Synthesis (RDNA -> Python) ---


def test_rdna_to_python_instruction() -> None:
  """
  Input: v_add_f32 v0, v1, v2
  Output: v0 = rdna.v_add_f32(v1, v2)
  """
  synth = RdnaSynthesizer(MagicMock())
  inst = Instruction("v_add_f32", [VGPR(0), VGPR(1), VGPR(2)])
  mod = synth.to_python([inst])
  code = mod.code
  assert "v0 = rdna.v_add_f32(v1, v2)" in code


def test_rdna_to_python_ranges() -> None:
  """
  Input: image_load v[0:3], v[4:7], s[0:3]
  Output: v_0_3 = rdna.image_load(v_4_7, s_0_3)
  """
  synth = RdnaSynthesizer(MagicMock())
  inst = Instruction("image_load", [VGPR(0, 4), VGPR(4, 4), SGPR(0, 4)])  # count=4 -> internal indices

  mod = synth.to_python([inst])
  code = mod.code

  # The allocator stringifies count > 1 as v[start:end].
  # Synthesizer `_convert_instruction_to_py` sanitizes brackets.
  # v[0:3] -> v_0_3
  assert "v_0_3 = rdna.image_load(v_4_7, s_0_3)" in code


def test_rdna_to_python_label() -> None:
  """
  Input: Label
  Output: Comment marker
  """
  synth = RdnaSynthesizer(MagicMock())
  nodes = [Label("L_LOOP")]
  mod = synth.to_python(nodes)
  code = mod.code
  # Should contain comment
  assert "# Label: L_LOOP" in code
