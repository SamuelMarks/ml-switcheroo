"""Test suite for the Differ module."""

from ml_switcheroo.core.compiler.differ import GraphDiffer, _is_likely_stateful, DeleteAction, ReplaceAction
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge


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
  assert isinstance(actions[0], DeleteAction)


def test_differ_replace_node():
  """Verifies the behavior of differ replace node."""
  differ = GraphDiffer()
  g1 = LogicalGraph(nodes=[LogicalNode("a", "Conv")], edges=[])
  g2 = LogicalGraph(nodes=[LogicalNode("fused_a", "Linear")], edges=[])
  actions = differ.diff(g1, g2)
  assert len(actions) == 2  # One init, one call, since Linear is stateful
  assert isinstance(actions[0], ReplaceAction)
  assert actions[0].is_init
  assert isinstance(actions[1], ReplaceAction)
  assert not actions[1].is_init


def test_differ_replace_node_stateless():
  """Verifies replace node that is stateless."""
  differ = GraphDiffer()
  g1 = LogicalGraph(nodes=[LogicalNode("a", "Conv")], edges=[])
  g2 = LogicalGraph(nodes=[LogicalNode("fused_a", "add")], edges=[])
  actions = differ.diff(g1, g2)
  assert len(actions) == 1
  assert isinstance(actions[0], ReplaceAction)
  assert not actions[0].is_init


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
  assert any(isinstance(a, DeleteAction) and a.node_id == "b" for a in actions)


def test_differ_unmatched_new():
  """Verifies behavior when new node doesn't match any anchor."""
  differ = GraphDiffer()
  g1 = LogicalGraph(nodes=[LogicalNode("a", "Conv")], edges=[])
  g2 = LogicalGraph(nodes=[LogicalNode("fused_b", "Linear")], edges=[])
  actions = differ.diff(g1, g2)
  assert len(actions) == 1
  assert isinstance(actions[0], DeleteAction)


def test__is_likely_stateful():
  """Verifies the behavior of is likely stateful."""
  assert _is_likely_stateful(LogicalNode("1", "Conv2d")) is True
  assert _is_likely_stateful(LogicalNode("2", "add")) is False
  assert _is_likely_stateful(LogicalNode("3", "fused_add")) is False
  assert _is_likely_stateful(LogicalNode("4", "my_FusedOp")) is True
  assert _is_likely_stateful(LogicalNode("5", "")) is False
  assert _is_likely_stateful(LogicalNode("6", None)) is False


def test_differ_diff_no_anchor():
  # Hit 123->128
  """Test differ diff no anchor."""
  from ml_switcheroo.core.compiler.differ import GraphDiffer
  from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode

  g1 = LogicalGraph("g1")
  g1.nodes.append(LogicalNode("A", "Op"))

  g2 = LogicalGraph("g2")
  g2.nodes.append(LogicalNode("B", "Op"))  # neither metadata anchor, nor starts with fused_
  # or starts with fused_ but candidate not in deleted_ids
  g2.nodes.append(LogicalNode("fused_C", "Op"))

  differ = GraphDiffer()
  # It will identify 'A' as deleted, 'B' and 'fused_C' as added.
  # When looping over new_nodes, it will check 'B' -> hits 123->128 (starts with fused == False)
  # Then 'fused_C' -> hits 123->128 ('C' not in deleted_ids)
  diffs = differ.diff(g1, g2)
  assert len(diffs) > 0
