"""
Tests for GraphOptimizer Feature via LogicalGraph.
"""

import pytest
from ml_switcheroo.core.graph_optimizer import GraphOptimizer
from ml_switcheroo.core.graph import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.core.dsl import PatternDef


@pytest.fixture
def patterns():
  return [
    PatternDef(name="CBR", sequence=["Conv2d", "BatchNorm", "ReLU"], replace_with="FusedCBR"),
    PatternDef(name="NormAct", sequence=["LayerNorm", "ReLU"], replace_with="FusedNormAct"),
  ]


@pytest.fixture
def graph():
  g = LogicalGraph()
  # input -> conv1 -> bn1 -> relu1 -> ln1 -> relu2 -> output
  g.nodes = [
    LogicalNode("in", "Input"),
    LogicalNode("c1", "Conv2d", {"k": "3"}),
    LogicalNode("b1", "BatchNorm", {"eps": "1e-5"}),
    LogicalNode("r1", "ReLU", {}),
    LogicalNode("l1", "LayerNorm", {"d": "64"}),
    LogicalNode("r2", "ReLU", {}),
    LogicalNode("out", "Output"),
  ]
  g.edges = [
    LogicalEdge("in", "c1"),
    LogicalEdge("c1", "b1"),
    LogicalEdge("b1", "r1"),
    LogicalEdge("r1", "l1"),
    LogicalEdge("l1", "r2"),
    LogicalEdge("r2", "out"),
  ]
  return g


def test_pattern_match_and_replace(graph, patterns):
  optimizer = GraphOptimizer(patterns)
  new_graph = optimizer.optimize(graph)

  # Expect c1, b1, r1 -> fused_c1 (FusedCBR)
  # Expect l1, r2 -> fused_l1 (FusedNormAct)
  # Total expected: 4 (in, fused_c1, fused_l1, out)
  assert len(new_graph.nodes) == 4

  node_ids = {n.id for n in new_graph.nodes}
  assert "in" in node_ids
  assert "out" in node_ids
  assert "fused_c1" in node_ids
  assert "fused_l1" in node_ids

  # 2. Check Meta Merge
  cbr = next(n for n in new_graph.nodes if n.id == "fused_c1")
  assert cbr.kind == "FusedCBR"
  assert cbr.metadata["k"] == "3"
  assert cbr.metadata["eps"] == "1e-5"

  # 3. Check Edges
  # in -> fused_c1 -> fused_l1 -> out
  edges = new_graph.edges
  assert len(edges) == 3
  assert edges[0].source == "in" and edges[0].target == "fused_c1"
  assert edges[1].source == "fused_c1" and edges[1].target == "fused_l1"
  assert edges[2].source == "fused_l1" and edges[2].target == "out"


def test_no_match(graph):
  # Empty patterns
  opt = GraphOptimizer([])
  res = opt.optimize(graph)
  assert len(res.nodes) == len(graph.nodes)


def test_partial_sequence_no_match():
  # Sequence [Conv, BN] but graph missing BN
  pat = [PatternDef(name="CB", sequence=["Conv2d", "BatchNorm"], replace_with="Fused")]

  g = LogicalGraph()
  g.nodes = [LogicalNode("c", "Conv2d"), LogicalNode("r", "ReLU")]
  g.edges = [LogicalEdge("c", "r")]

  opt = GraphOptimizer(pat)
  res = opt.optimize(g)

  # No change
  assert len(res.nodes) == 2
  assert res.nodes[0].kind == "Conv2d"
