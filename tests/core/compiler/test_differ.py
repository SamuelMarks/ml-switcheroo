"""Test suite for the Differ module."""

from ml_switcheroo.core.compiler.differ import DeleteAction, GraphDiffer, ReplaceAction, _is_likely_stateful
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode


def test_differ_no_changes() -> None:
  """Verify `GraphDiffer` returns `0` for differ no changes."""
  differ = GraphDiffer()
  g1 = LogicalGraph(nodes={n.id: n for n in [LogicalNode("a", "Conv")]}, edges=[])
  g2 = LogicalGraph(nodes={n.id: n for n in [LogicalNode("a", "Conv")]}, edges=[])
  assert len(differ.diff(g1, g2)) == 0


def test_differ_deleted_node() -> None:
  """Verify `GraphDiffer` handles differ deleted node as expected."""
  differ = GraphDiffer()
  g1 = LogicalGraph(nodes={n.id: n for n in [LogicalNode("a", "Conv")]}, edges=[])
  empty_nodes: list[LogicalNode] = []
  g2 = LogicalGraph(nodes={n.id: n for n in empty_nodes}, edges=[])
  actions = differ.diff(g1, g2)
  assert len(actions) > 0
  assert isinstance(actions[0], DeleteAction)


def test_differ_replace_node() -> None:
  """Verify `GraphDiffer` returns `2` for differ replace node."""
  differ = GraphDiffer()
  g1 = LogicalGraph(nodes={n.id: n for n in [LogicalNode("a", "Conv")]}, edges=[])
  g2 = LogicalGraph(nodes={n.id: n for n in [LogicalNode("fused_a", "Linear")]}, edges=[])
  actions = differ.diff(g1, g2)
  assert len(actions) == 2  # One init, one call, since Linear is stateful
  assert isinstance(actions[0], ReplaceAction)
  assert actions[0].is_init
  assert isinstance(actions[1], ReplaceAction)
  assert not actions[1].is_init


def test_differ_replace_node_stateless() -> None:
  """Verifies replace node that is stateless."""
  differ = GraphDiffer()
  g1 = LogicalGraph(nodes={n.id: n for n in [LogicalNode("a", "Conv")]}, edges=[])
  g2 = LogicalGraph(nodes={n.id: n for n in [LogicalNode("fused_a", "add")]}, edges=[])
  actions = differ.diff(g1, g2)
  assert len(actions) == 1
  assert isinstance(actions[0], ReplaceAction)
  assert not actions[0].is_init


def test_differ_insert_node() -> None:
  """Verify `GraphDiffer` returns `0` for differ insert node."""
  differ = GraphDiffer()
  empty_nodes: list[LogicalNode] = []
  g1 = LogicalGraph(nodes={n.id: n for n in empty_nodes}, edges=[])
  g2 = LogicalGraph(nodes={n.id: n for n in [LogicalNode("a", "Conv", attributes={"anchor": "missing"})]}, edges=[])
  actions = differ.diff(g1, g2)
  assert len(actions) == 0


def test_differ_complex_replace() -> None:
  """Verify `GraphDiffer` handles differ complex replace as expected."""
  differ = GraphDiffer()
  g1 = LogicalGraph(
    nodes={n.id: n for n in [LogicalNode("a", "Linear"), LogicalNode("b", "GELU")]}, edges=[LogicalEdge("a", "b")]
  )
  g2 = LogicalGraph(
    nodes={n.id: n for n in [LogicalNode("fused_a", "FusedLinearGELU", attributes={"anchor": "a"})]},
    edges=[LogicalEdge("x", "fused_a")],
  )
  actions = differ.diff(g1, g2)
  assert len(actions) > 0
  assert any(isinstance(a, DeleteAction) and a.node_id == "b" for a in actions)


def test_differ_unmatched_new() -> None:
  """Verifies behavior when new node doesn't match any anchor."""
  differ = GraphDiffer()
  g1 = LogicalGraph(nodes={n.id: n for n in [LogicalNode("a", "Conv")]}, edges=[])
  g2 = LogicalGraph(nodes={n.id: n for n in [LogicalNode("fused_b", "Linear")]}, edges=[])
  actions = differ.diff(g1, g2)
  assert len(actions) == 1
  assert isinstance(actions[0], DeleteAction)


def test__is_likely_stateful() -> None:
  """Verify `_is_likely_stateful` produces expected results across differ inputs."""
  assert _is_likely_stateful(LogicalNode("1", "Conv2d")) is True
  assert _is_likely_stateful(LogicalNode("2", "add")) is False
  assert _is_likely_stateful(LogicalNode("3", "fused_add")) is False
  assert _is_likely_stateful(LogicalNode("4", "my_FusedOp")) is True


def test_differ_diff_no_anchor() -> None:
  """Verify `LogicalGraph` handles differ diff no anchor as expected."""
  # Hit 123->128
  from ml_switcheroo.core.compiler.differ import GraphDiffer
  from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode

  g1 = LogicalGraph("g1")
  g1.add_node(LogicalNode("A", "Op"))

  g2 = LogicalGraph("g2")
  g2.add_node(LogicalNode("B", "Op"))  # neither metadata anchor, nor starts with fused_
  # or starts with fused_ but candidate not in deleted_ids
  g2.add_node(LogicalNode("fused_C", "Op"))

  differ = GraphDiffer()
  # It will identify 'A' as deleted, 'B' and 'fused_C' as added.
  # When looping over new_nodes, it will check 'B' -> hits 123->128 (starts with fused == False)
  # Then 'fused_C' -> hits 123->128 ('C' not in deleted_ids)
  diffs = differ.diff(g1, g2)
  assert len(diffs) > 0
