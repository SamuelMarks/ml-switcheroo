"""Docstring."""

from ml_switcheroo.core.graph_optimizer import GraphOptimizer
from ml_switcheroo.core.graph import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.core.dsl import PatternDef


def test_graph_optimizer_no_patterns():
  """Docstring."""
  graph = LogicalGraph(nodes=[LogicalNode(id="1", kind="A")], edges=[])
  opt = GraphOptimizer(patterns=[])
  opt_graph = opt.optimize(graph)
  assert len(opt_graph.nodes) == 1
  assert opt_graph.nodes[0].id == "1"


def test_graph_optimizer_simple_fusion():
  """Docstring."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="1", kind="Conv"),
      LogicalNode(id="2", kind="BatchNorm"),
      LogicalNode(id="3", kind="ReLU"),
    ],
    edges=[
      LogicalEdge(source="1", target="2"),
      LogicalEdge(source="2", target="3"),
    ],
  )
  pattern = PatternDef(name="ConvBNReLU", sequence=["Conv", "BatchNorm", "ReLU"], replace_with="FusedConv")
  opt = GraphOptimizer(patterns=[pattern])
  opt_graph = opt.optimize(graph)

  assert len(opt_graph.nodes) == 1
  assert opt_graph.nodes[0].kind == "FusedConv"
  assert opt_graph.nodes[0].id == "fused_1"
  assert len(opt_graph.edges) == 0


def test_graph_optimizer_fusion_with_surrounding_nodes():
  """Docstring."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="0", kind="Input"),
      LogicalNode(id="1", kind="Conv"),
      LogicalNode(id="2", kind="BatchNorm"),
      LogicalNode(id="3", kind="ReLU"),
      LogicalNode(id="4", kind="Output"),
    ],
    edges=[
      LogicalEdge(source="0", target="1"),
      LogicalEdge(source="1", target="2"),
      LogicalEdge(source="2", target="3"),
      LogicalEdge(source="3", target="4"),
    ],
  )
  pattern = PatternDef(name="ConvBNReLU", sequence=["Conv", "BatchNorm", "ReLU"], replace_with="FusedConv")
  opt = GraphOptimizer(patterns=[pattern])
  opt_graph = opt.optimize(graph)

  assert len(opt_graph.nodes) == 3
  assert set(n.id for n in opt_graph.nodes) == {"0", "fused_1", "4"}
  assert len(opt_graph.edges) == 2
  edge_pairs = set((e.source, e.target) for e in opt_graph.edges)
  assert edge_pairs == {("0", "fused_1"), ("fused_1", "4")}


def test_graph_optimizer_internal_edges_dropped():
  """Docstring."""
  # If there is an edge from Conv to Output directly (branching)
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="1", kind="Conv"),
      LogicalNode(id="2", kind="BatchNorm"),
      LogicalNode(id="3", kind="ReLU"),
      LogicalNode(id="4", kind="Output"),
    ],
    edges=[
      LogicalEdge(source="1", target="2"),
      LogicalEdge(source="2", target="3"),
      LogicalEdge(source="1", target="4"),  # Internal non-tail to non-fused
      LogicalEdge(source="0", target="2"),  # External to internal non-head
    ],
  )
  # also add node 0
  graph.nodes.insert(0, LogicalNode(id="0", kind="Input"))

  pattern = PatternDef(name="ConvBNReLU", sequence=["Conv", "BatchNorm", "ReLU"], replace_with="FusedConv")
  opt = GraphOptimizer(patterns=[pattern])
  opt_graph = opt.optimize(graph)

  # edge 1 -> 4 is dropped because 1 is internal non-tail
  # edge 0 -> 2 is dropped because 2 is internal non-head
  assert len(opt_graph.edges) == 0


def test_match_sequence_returns_none_empty_sequence():
  """Docstring."""
  opt = GraphOptimizer(patterns=[])
  res = opt._match_sequence(
    start_node=LogicalNode(id="1", kind="A"), sequence=[], node_map={}, out_edges={}, processed_ids=set()
  )
  assert res is None


def test_match_sequence_mismatch_first():
  """Docstring."""
  opt = GraphOptimizer(patterns=[])
  res = opt._match_sequence(
    start_node=LogicalNode(id="1", kind="B"), sequence=["A", "C"], node_map={}, out_edges={}, processed_ids=set()
  )
  assert res is None


def test_match_sequence_neighbor_not_found():
  """Docstring."""
  opt = GraphOptimizer(patterns=[])
  node_map = {"1": LogicalNode(id="1", kind="A"), "2": LogicalNode(id="2", kind="C")}
  res = opt._match_sequence(
    start_node=node_map["1"], sequence=["A", "B"], node_map=node_map, out_edges={"1": ["2"]}, processed_ids=set()
  )
  assert res is None


def test_match_sequence_neighbor_already_processed():
  """Docstring."""
  opt = GraphOptimizer(patterns=[])
  node_map = {"1": LogicalNode(id="1", kind="A"), "2": LogicalNode(id="2", kind="B")}
  res = opt._match_sequence(
    start_node=node_map["1"], sequence=["A", "B"], node_map=node_map, out_edges={"1": ["2"]}, processed_ids={"2"}
  )
  assert res is None
