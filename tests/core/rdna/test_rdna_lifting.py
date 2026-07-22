"""Test suite for the Rdna Lifting module."""

from typing import List
from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
from ml_switcheroo.core.compiler.frontends.rdna.nodes import Comment, Immediate, Instruction, RdnaNode, SGPR, VGPR


def make_inst(opcode: str, *operands) -> Instruction:
  """Helper to make inst."""
  return Instruction(opcode, list(operands))


def test_analyze_conv2d() -> None:
  """Analyzes conv2d."""
  s4 = SGPR(4)
  insts = [
    make_inst("s_mov_b32", s4, Immediate(0)),
    make_inst("s_cmp_lt_i32", s4, Immediate(3)),
    make_inst("s_cbranch_scc1"),
  ]
  meta = RdnaAnalyzer.analyze_block("Conv2d", insts)
  assert meta["k"] == 3
  assert meta["arg_2"] == 3


def test_analyze_linear() -> None:
  """Analyzes linear."""
  s0 = SGPR(0)
  insts = [make_inst("global_load_dword"), make_inst("s_cmp_lt_i32", s0, Immediate(512))]
  meta = RdnaAnalyzer.analyze_block("Linear", insts)
  assert meta["in_features"] == 512
  assert meta["arg_0"] == 512


def test_lift_simple_chain() -> None:
  """Lifts simple chain."""
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
  fc_node = next((n for n in graph.nodes if n.id == "fc1"))
  assert fc_node.metadata["in_features"] == 128


def test_lift_unmapped_op() -> None:
  """Lifts unmapped op."""
  nodes: List[RdnaNode] = [Comment("Input x -> v0"), Comment("Unmapped Op: torch.flatten (flat)"), Comment("Return: v0")]
  lifter = RdnaLifter()
  graph = lifter.lift(nodes)
  flat_node = next((n for n in graph.nodes if n.id == "flat"))
  assert flat_node.kind == "torch.flatten"
  assert flat_node.metadata["arg_1"] == 1


def test_lift_no_markers() -> None:
  """Lifts no markers."""
  nodes: List[RdnaNode] = [make_inst("v_add_f32", VGPR(0), VGPR(1), VGPR(2))]
  lifter = RdnaLifter()
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].kind == "rdna.v_add_f32"
