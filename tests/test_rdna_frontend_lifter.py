"""Test module."""

from typing import List

from ml_switcheroo.core.compiler.frontends.rdna.cst import (
  RdnaComment,
  RdnaImmediate,
  RdnaInstruction,
  RdnaModule,
  RdnaNode,
  c_SGPR,
)
from ml_switcheroo.core.compiler.frontends.rdna.lifter import RdnaLifter
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode


def test_rdna_lifter_basic() -> None:
  """Docstring."""
  lifter: RdnaLifter = RdnaLifter()

  # Manually inject a fake marker that isn't handled explicitly to hit the 'else' branch
  from ml_switcheroo.core.compiler.frontends.semantic_parser import SemanticMarker

  class FakeMarker(SemanticMarker):
    """Fake marker for testing fallback logic."""

    pass

  original_parse = lifter.comment_parser.parse

  def mock_parse(text):
    if "FAKE_MARKER" in text:
      return FakeMarker()
    return original_parse(text)

  lifter.comment_parser.parse = mock_parse

  nodes: List[RdnaNode] = [
    RdnaComment(text="; FAKE_MARKER"),
    RdnaComment(text="; Input x ->"),
    RdnaComment(text="; BEGIN Conv2d(block_1)"),
    RdnaInstruction(opcode="s_cmp_lt_i32", operands=[c_SGPR(0), RdnaImmediate(value=3)]),
    RdnaComment(text="; END Conv2d(block_1)"),
    RdnaComment(text="; Unmapped Op: flatten(flatten)"),
    RdnaComment(text="; Return:"),
    RdnaComment(text="; Return:"),  # Duplicate for seen_ids branch
    RdnaComment(text="; Unmapped Op: unknown(other)"),  # Unmapped branch without flatten
    RdnaInstruction(opcode="v_add_f32", operands=[]),  # implicit block hit
  ]
  graph: LogicalGraph = lifter.lift(nodes)

  assert isinstance(graph, LogicalGraph)
  assert len(graph.nodes) > 0

  node_ids: List[str] = [n.id for n in graph.nodes]
  assert "x" in node_ids
  assert "block_1" in node_ids
  assert "flatten" in node_ids
  assert "output" in node_ids

  # check if flatten has arg_1=1
  flatten_node: LogicalNode = next(n for n in graph.nodes if n.id == "flatten")
  assert flatten_node.metadata.get("arg_1") == 1

  # check if Conv2d has k=3
  conv_node: LogicalNode = next(n for n in graph.nodes if n.id == "block_1")
  assert conv_node.metadata.get("k") == 3


def test_rdna_lifter_seen_ids() -> None:
  """Docstring."""
  lifter: RdnaLifter = RdnaLifter()
  nodes: List[RdnaNode] = [
    RdnaComment(text="; Input x ->"),
    RdnaComment(text="; Input x ->"),  # Duplicate ID
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 1


def test_rdna_lifter_instruction() -> None:
  """Docstring."""
  lifter: RdnaLifter = RdnaLifter()
  nodes: List[RdnaNode] = [RdnaInstruction(opcode="v_add_f32", operands=[])]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "inst_0"


def test_rdna_lifter_multiple_return() -> None:
  """Docstring."""
  lifter: RdnaLifter = RdnaLifter()
  nodes: List[RdnaNode] = [
    RdnaComment(text="; Input x ->"),
    RdnaComment(text="; Return:"),
    RdnaComment(text="; Return:"),  # Output already in seen_ids
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  node_ids: List[str] = [n.id for n in graph.nodes]
  assert "x" in node_ids
  assert "output" in node_ids
  assert len(graph.nodes) == 2


def test_rdna_lifter_unmapped_other() -> None:
  """Docstring."""
  lifter: RdnaLifter = RdnaLifter()
  nodes: List[RdnaNode] = [
    RdnaComment(text="; Unmapped Op: other_op(other_op)"),
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  node_ids: List[str] = [n.id for n in graph.nodes]
  assert "other_op" in node_ids
  other_node: LogicalNode = next(n for n in graph.nodes if n.id == "other_op")
  assert "arg_1" not in other_node.metadata


def test_rdna_lifter_return_no_previous() -> None:
  """Docstring."""
  lifter: RdnaLifter = RdnaLifter()
  nodes: List[RdnaNode] = [
    RdnaComment(text="; Return:"),
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert graph.nodes[0].id == "output"
  assert len(graph.edges) == 0


def test_rdna_lifter_comment_unparsed() -> None:
  """Docstring."""
  lifter: RdnaLifter = RdnaLifter()
  nodes: List[RdnaNode] = [
    RdnaComment(text="; JUST A NORMAL COMMENT"),
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


# --- Merged from test_rdna_frontend_lifter_extra.py ---


def test_lifter_end_mismatch() -> None:
  """Docstring."""
  mod: RdnaModule = RdnaModule(
    statements=[
      RdnaComment(text="; BEGIN Linear(some_id)"),
      RdnaComment(text="; END Linear(some_other_id)"),  # mismatch
    ]
  )
  lifter: RdnaLifter = RdnaLifter()
  graph: LogicalGraph = lifter.lift(mod.statements)
  assert graph is not None


def test_lifter_return_seen() -> None:
  """Docstring."""
  mod: RdnaModule = RdnaModule(
    statements=[
      RdnaComment(text="; RETURN"),
      RdnaComment(text="; RETURN"),
    ]
  )
  lifter: RdnaLifter = RdnaLifter()
  lifter.lift(mod.statements)
