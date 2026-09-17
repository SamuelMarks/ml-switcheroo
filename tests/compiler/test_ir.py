"""Test suite for the Ir module."""

from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode, topological_sort


def test_graph_structure() -> None:
  """Verifies the behavior of graph structure."""
  node_a = LogicalNode(id="a", op_type="Op", attributes={"k": "3"})
  node_b = LogicalNode(id="b", op_type="Op")
  assert node_a.id == "a"
  assert node_a.op_type == "Op"
  assert node_a.attributes["k"] == "3"
  edge = LogicalEdge(source="a", target="b")
  assert edge.source == "a"
  assert edge.target == "b"
  graph = LogicalGraph(nodes={"a": node_a, "b": node_b}, edges=[edge])
  assert len(graph.nodes) == 2
  assert len(graph.edges) == 1


def test_topological_sort_linear() -> None:
  """Verifies the behavior of topological sort linear."""
  nodes = {
    "c": LogicalNode(id="c", op_type="op"),
    "a": LogicalNode(id="a", op_type="op"),
    "b": LogicalNode(id="b", op_type="op"),
  }
  edges = [LogicalEdge("a", "b"), LogicalEdge("b", "c")]
  g = LogicalGraph(nodes=nodes, edges=edges)
  sorted_nodes: list[LogicalNode] = topological_sort(g)
  ids: list[str] = [n.id for n in sorted_nodes]
  assert ids == ["a", "b", "c"]


def test_topological_sort_branch() -> None:
  """Verifies the behavior of topological sort branch."""
  nodes = {
    "a": LogicalNode("a", op_type="op"),
    "b": LogicalNode("b", op_type="op"),
    "c": LogicalNode("c", op_type="op"),
  }
  edges = [LogicalEdge("a", "b"), LogicalEdge("a", "c")]
  g = LogicalGraph(nodes=nodes, edges=edges)
  sorted_nodes: list[LogicalNode] = topological_sort(g)
  ids: list[str] = [n.id for n in sorted_nodes]
  assert ids[0] == "a"
  assert "b" in ids[1:]
  assert "c" in ids[1:]


def test_topological_sort_cycle_resilience() -> None:
  """Verifies the behavior of topological sort cycle resilience."""
  nodes = {
    "a": LogicalNode("a", op_type="op"),
    "b": LogicalNode("b", op_type="op"),
  }
  edges = [LogicalEdge("a", "b"), LogicalEdge("b", "a")]
  g = LogicalGraph(nodes=nodes, edges=edges)
  sorted_nodes: list[LogicalNode] = topological_sort(g)
  assert len(sorted_nodes) == 2


def test_logical_axis() -> None:
  """Verifies the behavior of logical axis."""
  from ml_switcheroo.core.compiler.ir import LogicalAxis

  axis = LogicalAxis(name="embed", size=1024)
  assert axis.name == "embed"
  assert axis.size == 1024


def test_partition_spec() -> None:
  """Verifies the behavior of partition spec."""
  from ml_switcheroo.core.compiler.ir import PartitionSpec

  spec = PartitionSpec(axes=("data", None, ("model", "tensor")))
  assert len(spec.axes) == 3
  assert spec.axes[0] == "data"
  assert spec.axes[1] is None
  assert spec.axes[2] == ("model", "tensor")


def test_logical_mesh() -> None:
  """Verifies the behavior of logical mesh."""
  from ml_switcheroo.core.compiler.ir import LogicalMesh

  mesh = LogicalMesh(shape={"data": 2, "model": 4})
  assert mesh.shape["data"] == 2
  assert mesh.shape["model"] == 4


def test_graph_sharding_attributes() -> None:
  """Verifies the behavior of graph sharding attributes."""
  from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalMesh, LogicalNode, PartitionSpec

  mesh = LogicalMesh(shape={"data": 8})
  spec = PartitionSpec(axes=("data", None))
  node = LogicalNode(id="sharded_node", op_type="Op", sharding=spec)
  graph = LogicalGraph(nodes={"sharded_node": node}, mesh=mesh)
  assert graph.mesh is not None
  assert graph.mesh.shape["data"] == 8
  assert graph.nodes["sharded_node"].sharding is not None
  assert graph.nodes["sharded_node"].sharding.axes == ("data", None)


def test_topological_sort_missing_node() -> None:
  """Verifies the behavior when an edge points to a non-existent node."""
  n_a = LogicalNode(id="a", op_type="op")
  # edge points to "b" which is not in nodes
  g = LogicalGraph(nodes={"a": n_a}, edges=[LogicalEdge("a", "b")])
  sorted_nodes: list[LogicalNode] = topological_sort(g)
  assert len(sorted_nodes) == 1
  assert sorted_nodes[0].id == "a"


def test_ir_topological_sort_cycle() -> None:
  """Docstring."""
  from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode, topological_sort

  nodes = {
    "A": LogicalNode("A", op_type="Op"),
    "B": LogicalNode("B", op_type="Op"),
    "C": LogicalNode("C", op_type="Op"),
  }
  # A -> B, B -> C, C -> B (cycle)
  edges = [
    LogicalEdge("A", "B"),
    LogicalEdge("B", "C"),
    LogicalEdge("C", "B"),
  ]
  g = LogicalGraph("Test", nodes=nodes, edges=edges)
  sorted_nodes: list[LogicalNode] = topological_sort(g)
  assert len(sorted_nodes) == 3
