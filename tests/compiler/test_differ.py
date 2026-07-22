"""Test suite for the Differ module."""

from ml_switcheroo.core.compiler.differ import GraphDiffer, _is_likely_stateful
from ml_switcheroo.core.graph import LogicalGraph, LogicalNode, LogicalEdge


def test_differ_no_changes():
  """Verifies the behavior of differ no changes."""
  differ = GraphDiffer()
  g1 = LogicalGraph(nodes=[LogicalNode("a", "Conv")], edges=[])
  g2 = LogicalGraph(nodes=[LogicalNode("a", "Conv")], edges=[])
  assert len(differ.diff(g1, g2)) == 0


def test_differ_deleted_node():
  """Verifies the behavior of differ deleted node."""
  differ = GraphDiffer()
  g1 = LogicalGraph(nodes=[LogicalNode("a", "Conv")], edges=[])
  g2 = LogicalGraph(nodes=[], edges=[])
  actions = differ.diff(g1, g2)
  assert len(actions) > 0
  assert actions[0].__class__.__name__ == "DeleteAction"


def test_differ_replace_node():
  """Verifies the behavior of differ replace node."""
  differ = GraphDiffer()
  g1 = LogicalGraph(nodes=[LogicalNode("a", "Conv")], edges=[])
  g2 = LogicalGraph(nodes=[LogicalNode("fused_a", "Linear")], edges=[])
  actions = differ.diff(g1, g2)
  assert len(actions) > 0
  assert actions[0].__class__.__name__ == "ReplaceAction"


def test_differ_insert_node():
  """Verifies the behavior of differ insert node."""
  differ = GraphDiffer()
  g1 = LogicalGraph(nodes=[], edges=[])
  g2 = LogicalGraph(nodes=[LogicalNode("a", "Conv", metadata={"anchor": "missing"})], edges=[])
  actions = differ.diff(g1, g2)
  assert len(actions) == 0


def test_differ_complex_replace():
  """Verifies the behavior of differ complex replace."""
  differ = GraphDiffer()
  g1 = LogicalGraph(nodes=[LogicalNode("a", "Linear"), LogicalNode("b", "GELU")], edges=[LogicalEdge("a", "b")])
  g2 = LogicalGraph(
    nodes=[LogicalNode("fused_a", "FusedLinearGELU", metadata={"anchor": "a"})], edges=[LogicalEdge("x", "fused_a")]
  )
  actions = differ.diff(g1, g2)
  assert len(actions) > 0


def test__is_likely_stateful():
  """Verifies the behavior of is likely stateful."""
  assert _is_likely_stateful(LogicalNode("1", "Conv2d")) is True
  assert _is_likely_stateful(LogicalNode("2", "add")) is False
  assert _is_likely_stateful(LogicalNode("3", "fused_add")) is False
  assert _is_likely_stateful(LogicalNode("4", "my_FusedOp")) is True
  assert _is_likely_stateful(LogicalNode("5", "")) is False
