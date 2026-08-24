"""Test suite for the Rdna Lifting module."""

from typing import List
from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaImmediate,
  RdnaInstruction,
  RdnaNode,
  RdnaSGPR,
  RdnaVGPR,
)


def make_inst(opcode: str, *operands) -> RdnaInstruction:
  """Helper to make inst."""
  return RdnaInstruction(opcode=opcode, operands=list(operands))


def test_analyze_conv2d() -> None:
  """Analyzes conv2d."""
  s4 = RdnaSGPR(index=4)
  insts = [
    make_inst("s_mov_b32", s4, RdnaImmediate(value=0)),
    make_inst("s_cmp_lt_i32", s4, RdnaImmediate(value=3)),
    make_inst("s_cbranch_scc1"),
  ]
  meta = RdnaAnalyzer.analyze_block("Conv2d", insts)
  assert meta["k"] == 3
  assert meta["arg_2"] == 3


def test_analyze_linear() -> None:
  """Analyzes linear."""
  s0 = RdnaSGPR(index=0)
  insts = [make_inst("global_load_dword"), make_inst("s_cmp_lt_i32", s0, RdnaImmediate(value=512))]
  meta = RdnaAnalyzer.analyze_block("Linear", insts)
  assert meta["in_features"] == 512
  assert meta["arg_0"] == 512


def test_lift_simple_chain() -> None:
  """Lifts simple chain."""
  nodes: List[RdnaNode] = [
    RdnaComment(text="Input x -> v0"),
    RdnaComment(text="BEGIN Linear (fc1)"),
    make_inst("s_cmp_lt_i32", RdnaSGPR(index=0), RdnaImmediate(value=128)),
    RdnaComment(text="END Linear (fc1)"),
    RdnaComment(text="Return: v10"),
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
  nodes: List[RdnaNode] = [
    RdnaComment(text="Input x -> v0"),
    RdnaComment(text="Unmapped Op: torch.flatten (flat)"),
    RdnaComment(text="Return: v0"),
  ]
  lifter = RdnaLifter()
  graph = lifter.lift(nodes)
  flat_node = next((n for n in graph.nodes if n.id == "flat"))
  assert flat_node.kind == "torch.flatten"
  assert flat_node.metadata["arg_1"] == 1


def test_lift_no_markers() -> None:
  """Lifts no markers."""
  nodes: List[RdnaNode] = [make_inst("v_add_f32", RdnaVGPR(index=0), RdnaVGPR(index=1), RdnaVGPR(index=2))]
  lifter = RdnaLifter()
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].kind == "rdna.v_add_f32"


def test_rdna_analysis_conv2d_fallback():
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction, RdnaImmediate

  # We need loop_limits to be populated. The code checks for s_cmp_lt_i32
  inst = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaImmediate(value=3)])

  meta = RdnaAnalyzer.analyze_block("Conv2d", [inst])
  assert meta["k"] == 3


def test_rdna_analysis_linear_fallback():
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction, RdnaImmediate

  inst = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaImmediate(value=10)])

  meta = RdnaAnalyzer.analyze_block("Linear", [inst])
  assert meta["in_features"] == 10


def test_rdna_analysis_unknown_kind():
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction, RdnaImmediate

  inst = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaImmediate(value=10)])

  meta = RdnaAnalyzer.analyze_block("UnknownKind", [inst])
  assert meta == {}


def test_rdna_analysis_no_loop_limits():
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.analysis import RdnaAnalyzer
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaInstruction, RdnaSGPR

  inst = RdnaInstruction(opcode="s_cmp_lt_i32", operands=[RdnaSGPR(index=0)])

  meta = RdnaAnalyzer.analyze_block("Conv2d", [inst])
  assert meta == {}


def test_rdna_lifter_seen_ids():
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment

  lifter = RdnaLifter()
  nodes = [
    RdnaComment(text="; BEGIN Add (Add_0)"),
    RdnaComment(text="; END Add (Add_0)"),
    RdnaComment(text="; BEGIN Add (Add_0)"),
    RdnaComment(text="; END Add (Add_0)"),
  ]

  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1


def test_rdna_lifter_unmapped():
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment

  lifter = RdnaLifter()
  nodes = [RdnaComment(text="; Unmapped Op: flatten (flatten_0)")]

  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].metadata.get("arg_1") == 1


def test_rdna_lifter_input():
  """Docstring."""
  from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
  from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment

  lifter = RdnaLifter()
  nodes = [RdnaComment(text="; Input arg_0 ->")]

  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].kind == "Input"
