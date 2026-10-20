"""Test suite for rdna frontend lifter."""

import typing

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
  """Verify rdna lifter basic behavior."""
  lifter: RdnaLifter = RdnaLifter()

  # Manually inject a fake marker that isn't handled explicitly to hit the 'else' branch
  from ml_switcheroo.core.compiler.frontends.semantic_parser import SemanticMarker

  class FakeMarker(SemanticMarker):
    """Fake marker for testing fallback logic."""

    pass

  original_parse = lifter.comment_parser.parse

  def mock_parse(text: str) -> typing.Optional[SemanticMarker]:
    """Execute mock_parse test helper.

    Args:
        text: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if "FAKE_MARKER" in text:
      return FakeMarker()
    return original_parse(text)

  setattr(lifter.comment_parser, "parse", mock_parse)

  nodes: typing.List[RdnaNode] = [
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

  node_ids: typing.List[str] = [n.id for n in graph.nodes.values()]
  assert "x" in node_ids
  assert "block_1" in node_ids
  assert "flatten" in node_ids
  assert "output" in node_ids

  # check if flatten has arg_1=1
  flatten_node: LogicalNode = graph.nodes["flatten"]
  assert flatten_node.attributes.get("arg_1") == 1

  # check if Conv2d has k=3
  conv_node: LogicalNode = graph.nodes["block_1"]
  assert conv_node.attributes.get("k") == 3


def test_rdna_lifter_seen_ids() -> None:
  """Verify rdna lifter seen ids behavior."""
  lifter: RdnaLifter = RdnaLifter()
  nodes: typing.List[RdnaNode] = [
    RdnaComment(text="; Input x ->"),
    RdnaComment(text="; Input x ->"),  # Duplicate ID
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 1


def test_rdna_lifter_instruction() -> None:
  """Verify rdna lifter instruction behavior."""
  lifter: RdnaLifter = RdnaLifter()
  nodes: typing.List[RdnaNode] = [RdnaInstruction(opcode="v_add_f32", operands=[])]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert "inst_0" in graph.nodes


def test_rdna_lifter_multiple_return() -> None:
  """Verify rdna lifter multiple return behavior."""
  lifter: RdnaLifter = RdnaLifter()
  nodes: typing.List[RdnaNode] = [
    RdnaComment(text="; Input x ->"),
    RdnaComment(text="; Return:"),
    RdnaComment(text="; Return:"),  # Output already in seen_ids
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  node_ids: typing.List[str] = list(graph.nodes.keys())
  assert "x" in node_ids
  assert "output" in node_ids
  assert len(graph.nodes) == 2


def test_rdna_lifter_unmapped_other() -> None:
  """Verify rdna lifter unmapped other behavior."""
  lifter: RdnaLifter = RdnaLifter()
  nodes: typing.List[RdnaNode] = [
    RdnaComment(text="; Unmapped Op: other_op(other_op)"),
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  node_ids: typing.List[str] = list(graph.nodes.keys())
  assert "other_op" in node_ids
  other_node: LogicalNode = graph.nodes["other_op"]
  assert "arg_1" not in other_node.attributes


def test_rdna_lifter_return_no_previous() -> None:
  """Verify rdna lifter return no previous behavior."""
  lifter: RdnaLifter = RdnaLifter()
  nodes: typing.List[RdnaNode] = [
    RdnaComment(text="; Return:"),
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 1
  assert "output" in graph.nodes
  assert len(graph.edges) == 0


def test_rdna_lifter_comment_unparsed() -> None:
  """Verify rdna lifter comment unparsed behavior."""
  lifter: RdnaLifter = RdnaLifter()
  nodes: typing.List[RdnaNode] = [
    RdnaComment(text="; JUST A NORMAL COMMENT"),
  ]
  graph: LogicalGraph = lifter.lift(nodes)
  assert len(graph.nodes) == 0


# --- Merged from test_rdna_frontend_lifter_extra.py ---


def test_lifter_end_mismatch() -> None:
  """Verify lifter end mismatch behavior."""
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
  """Verify lifter return seen behavior."""
  mod: RdnaModule = RdnaModule(
    statements=[
      RdnaComment(text="Return: v0"),
      RdnaComment(text="Return: v0"),
    ]
  )
  lifter: RdnaLifter = RdnaLifter()
  _res = lifter.lift(mod.statements)
  assert "output" in _res.nodes
  assert _res.nodes["output"].op_type == "Output"
