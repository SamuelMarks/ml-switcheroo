"""
Tests for the RDNA Lifter analysis logic.

Verifies that RDNA comments are correctly parsed into a LogicalGraph structure
reflecting the original model, including parameter recovery via analysis.
"""

from typing import List

from ml_switcheroo.core.graph import LogicalGraph
from ml_switcheroo.core.rdna.analysis import RdnaAnalyzer
from ml_switcheroo.core.rdna.lifter import RdnaLifter
from ml_switcheroo.core.rdna.nodes import (
  Comment,
  Immediate,
  Instruction,
  RdnaNode,
  SGPR,
  VGPR,
)


def make_inst(opcode: str, *operands) -> Instruction:
  """Helper to build instruction."""
  return Instruction(opcode, list(operands))


def test_analyze_conv2d() -> None:
  """
  Verify analyzer extracts kernel size from s_cmp_lt_i32.
  """
  s4 = SGPR(4)
  insts = [
    make_inst("s_mov_b32", s4, Immediate(0)),
    make_inst("s_cmp_lt_i32", s4, Immediate(3)),  # Kernel Size
    make_inst("s_cbranch_scc1"),
  ]

  meta = RdnaAnalyzer.analyze_block("Conv2d", insts)
  assert meta["k"] == 3
  assert meta["arg_2"] == 3


def test_analyze_linear() -> None:
  """
  Verify analyzer extracts in_features.
  """
  s0 = SGPR(0)
  insts = [
    make_inst("global_load_dword"),
    make_inst("s_cmp_lt_i32", s0, Immediate(512)),  # Limit
  ]

  meta = RdnaAnalyzer.analyze_block("Linear", insts)
  assert meta["in_features"] == 512
  assert meta["arg_0"] == 512


def test_lift_simple_chain() -> None:
  """
  Scenario: Input -> Linear -> Output.
  """
  nodes: List[RdnaNode] = [
    Comment("Input x -> v0"),
    Comment("BEGIN Linear (fc1)"),
    make_inst("s_cmp_lt_i32", SGPR(0), Immediate(128)),
    Comment("END Linear (fc1)"),
    Comment("Return: v10"),
  ]

  lifter = RdnaLifter()
  graph = lifter.lift(nodes)

  assert len(graph.nodes) == 3
  ids = [n.id for n in graph.nodes]
  assert ids == ["x", "fc1", "output"]

  # Check Metadata
  fc_node = next(n for n in graph.nodes if n.id == "fc1")
  assert fc_node.metadata["in_features"] == 128


def test_lift_unmapped_op() -> None:
  """
  Scenario: Unknown op preserved as comment.
  """
  nodes: List[RdnaNode] = [
    Comment("Input x -> v0"),
    Comment("Unmapped Op: torch.flatten (flat)"),
    Comment("Return: v0"),
  ]

  lifter = RdnaLifter()
  graph = lifter.lift(nodes)

  flat_node = next(n for n in graph.nodes if n.id == "flat")
  assert flat_node.kind == "torch.flatten"
  # Flatten default arg heuristic
  assert flat_node.metadata["arg_1"] == 1


def test_lift_no_markers() -> None:
  """
  Scenario: Raw assembly without semantic comments.
  Expect: Empty graph.
  """
  nodes: List[RdnaNode] = [make_inst("v_add_f32", VGPR(0), VGPR(1), VGPR(2))]
  lifter = RdnaLifter()
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 0
