import pytest
from ml_switcheroo.compiler.differ import GraphDiffer, _is_likely_stateful
from ml_switcheroo.core.graph import LogicalGraph, LogicalNode, LogicalEdge


def test_differ_no_changes():
  differ = GraphDiffer()
  g1 = LogicalGraph(nodes=[LogicalNode("a", "Conv")], edges=[])
  g2 = LogicalGraph(nodes=[LogicalNode("a", "Conv")], edges=[])
  assert len(differ.diff(g1, g2)) == 0


def test_differ_deleted_node():
  differ = GraphDiffer()
  g1 = LogicalGraph(nodes=[LogicalNode("a", "Conv")], edges=[])
  g2 = LogicalGraph(nodes=[], edges=[])
  actions = differ.diff(g1, g2)
  assert len(actions) > 0
  assert actions[0].__class__.__name__ == "DeleteAction"


def test_differ_replace_node():
  differ = GraphDiffer()
  # Replace single node
  g1 = LogicalGraph(nodes=[LogicalNode("a", "Conv")], edges=[])
  g2 = LogicalGraph(nodes=[LogicalNode("fused_a", "Linear")], edges=[])
  actions = differ.diff(g1, g2)
  assert len(actions) > 0
  assert actions[0].__class__.__name__ == "ReplaceAction"


def test_differ_insert_node():
  differ = GraphDiffer()
  g1 = LogicalGraph(nodes=[], edges=[])
  g2 = LogicalGraph(nodes=[LogicalNode("a", "Conv", metadata={"anchor": "missing"})], edges=[])
  actions = differ.diff(g1, g2)
  # Since "missing" is not in deleted_ids, it doesnt match anchor. So it is ignored right now.
  assert len(actions) == 0


def test_differ_complex_replace():
  differ = GraphDiffer()
  # 2 nodes become 1 node (Fusion)
  g1 = LogicalGraph(nodes=[LogicalNode("a", "Linear"), LogicalNode("b", "GELU")], edges=[LogicalEdge("a", "b")])
  # The new node takes the place of both. For fusion, `GraphOptimizer` usually retains the ID of the first node or creates a new one.
  g2 = LogicalGraph(
    nodes=[LogicalNode("fused_a", "FusedLinearGELU", metadata={"anchor": "a"})], edges=[LogicalEdge("x", "fused_a")]
  )
  actions = differ.diff(g1, g2)
  # The differ heuristics match common inputs/outputs.
  # Since g1 and g2 have different IDs and no shared context, it treats it as ReplaceAction
  assert len(actions) > 0


def test__is_likely_stateful():
  assert _is_likely_stateful(LogicalNode("1", "Conv2d")) is True
  assert _is_likely_stateful(LogicalNode("2", "add")) is False
  assert _is_likely_stateful(LogicalNode("3", "fused_add")) is False  # wait
  assert _is_likely_stateful(LogicalNode("4", "my_FusedOp")) is True
  assert _is_likely_stateful(LogicalNode("5", "")) is False  # check empty string guard
