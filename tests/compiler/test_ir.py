"""
Tests for IR Data Structures and Algorithms.

Verifies:
1. LogicalNode/Edge creation.
2. Topological Sort logic correctness and cycle handling.
"""

from ml_switcheroo.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge, topological_sort


def test_graph_structure():
  """Verify IR data structures."""
  node = LogicalNode(id="test", kind="Op", metadata={"k": "3"})
  assert node.id == "test"
  assert node.kind == "Op"
  assert node.metadata["k"] == "3"

  edge = LogicalEdge(source="a", target="b")
  assert edge.source == "a"
  assert edge.target == "b"

  graph = LogicalGraph(nodes=[node], edges=[edge])
  assert len(graph.nodes) == 1
  assert len(graph.edges) == 1


def test_topological_sort_linear():
  """Verify sorting of a linear chain A->B->C."""
  g = LogicalGraph()
  n_a = LogicalNode(id="a", kind="op")
  n_b = LogicalNode(id="b", kind="op")
  n_c = LogicalNode(id="c", kind="op")
  g.nodes = [n_c, n_a, n_b]  # Random order
  g.edges = [LogicalEdge("a", "b"), LogicalEdge("b", "c")]

  sorted_nodes = topological_sort(g)
  ids = [n.id for n in sorted_nodes]
  assert ids == ["a", "b", "c"]


def test_topological_sort_branch():
  """Verify sorting of A->B, A->C."""
  g = LogicalGraph()
  n_a = LogicalNode("a", "op")
  n_b = LogicalNode("b", "op")
  n_c = LogicalNode("c", "op")
  g.nodes = [n_a, n_b, n_c]
  g.edges = [LogicalEdge("a", "b"), LogicalEdge("a", "c")]

  sorted_nodes = topological_sort(g)
  ids = [n.id for n in sorted_nodes]
  assert ids[0] == "a"
  assert "b" in ids[1:]
  assert "c" in ids[1:]


def test_topological_sort_cycle_resilience():
  """Verify cycles don't crash and nodes are preserved."""
  g = LogicalGraph()
  n_a = LogicalNode("a", "op")
  n_b = LogicalNode("b", "op")
  g.nodes = [n_a, n_b]
  g.edges = [LogicalEdge("a", "b"), LogicalEdge("b", "a")]

  # Should fall back to definition order or partial sort
  sorted_nodes = topological_sort(g)
  assert len(sorted_nodes) == 2
