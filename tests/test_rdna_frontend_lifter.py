"""Test module."""

from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaComment, RdnaInstruction, RdnaImmediate, c_SGPR
from ml_switcheroo.core.compiler.ir import LogicalGraph


def test_rdna_lifter_basic():
  """Test element."""
  lifter = RdnaLifter()
  nodes = [
    RdnaComment(text="; Input x ->"),
    RdnaComment(text="; BEGIN Conv2d(block_1)"),
    RdnaInstruction(opcode="s_cmp_lt_i32", operands=[c_SGPR(0), RdnaImmediate(value=3)]),
    RdnaComment(text="; END Conv2d(block_1)"),
    RdnaComment(text="; Unmapped Op: flatten(flatten)"),
    RdnaComment(text="; Return:"),
  ]
  graph = lifter.lift(nodes)

  assert isinstance(graph, LogicalGraph)
  assert len(graph.nodes) > 0

  node_ids = [n.id for n in graph.nodes]
  assert "x" in node_ids
  assert "block_1" in node_ids
  assert "flatten" in node_ids
  assert "output" in node_ids

  # check if flatten has arg_1=1
  flatten_node = next(n for n in graph.nodes if n.id == "flatten")
  assert flatten_node.metadata.get("arg_1") == 1

  # check if Conv2d has k=3
  conv_node = next(n for n in graph.nodes if n.id == "block_1")
  assert conv_node.metadata.get("k") == 3


def test_rdna_lifter_seen_ids():
  """Test element."""
  lifter = RdnaLifter()
  nodes = [
    RdnaComment(text="; Input x ->"),
    RdnaComment(text="; Input x ->"),  # Duplicate ID
  ]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1


def test_rdna_lifter_instruction():
  """Test element."""
  lifter = RdnaLifter()
  nodes = [RdnaInstruction(opcode="v_add_f32", operands=[])]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "inst_0"


def test_rdna_lifter_multiple_return():
  """Test element."""
  lifter = RdnaLifter()
  nodes = [
    RdnaComment(text="; Input x ->"),
    RdnaComment(text="; Return:"),
    RdnaComment(text="; Return:"),  # Output already in seen_ids
  ]
  graph = lifter.lift(nodes)
  node_ids = [n.id for n in graph.nodes]
  assert "x" in node_ids
  assert "output" in node_ids
  assert len(graph.nodes) == 2


def test_rdna_lifter_unmapped_other():
  """Test element."""
  lifter = RdnaLifter()
  nodes = [
    RdnaComment(text="; Unmapped Op: other_op(other_op)"),
  ]
  graph = lifter.lift(nodes)
  node_ids = [n.id for n in graph.nodes]
  assert "other_op" in node_ids
  other_node = next(n for n in graph.nodes if n.id == "other_op")
  assert "arg_1" not in other_node.metadata


def test_rdna_lifter_return_no_previous():
  """Test element."""
  lifter = RdnaLifter()
  nodes = [
    RdnaComment(text="; Return:"),
  ]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "output"
  assert len(graph.edges) == 0


def test_rdna_lifter_comment_unparsed():
  """Test element."""
  lifter = RdnaLifter()
  nodes = [
    RdnaComment(text="; JUST A NORMAL COMMENT"),
  ]
  graph = lifter.lift(nodes)
  assert len(graph.nodes) == 0
