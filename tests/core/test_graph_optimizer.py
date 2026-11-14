"""Unit tests for `GraphOptimizer` fusion sequences (`ml_switcheroo.core.graph_optimizer`)."""

from ml_switcheroo.core.dsl import PatternDef
from ml_switcheroo.core.graph import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.graph_optimizer import GraphOptimizer


def build_test_graph() -> LogicalGraph:
  """Construct a standard `Input -> Conv2d -> BatchNorm -> ReLU -> Output` test graph.

  Returns:
      Populated 5-node `LogicalGraph`.
  """
  nodes = {
    "in": LogicalNode("in", op_type="Input"),
    "conv": LogicalNode("conv", op_type="Conv2d", attributes={"arg_0": "3", "arg_1": "16"}),
    "bn": LogicalNode("bn", op_type="BatchNorm", attributes={"eps": "1e-5"}),
    "relu": LogicalNode("relu", op_type="ReLU"),
    "out": LogicalNode("out", op_type="Output"),
  }
  edges = [
    LogicalEdge("in", "conv"),
    LogicalEdge("conv", "bn"),
    LogicalEdge("bn", "relu"),
    LogicalEdge("relu", "out"),
  ]
  return LogicalGraph(nodes=nodes, edges=edges)


def test_graph_optimizer_no_patterns_and_no_match() -> None:
  """Verify `GraphOptimizer.optimize` returns unchanged topology when patterns are empty or non-matching."""
  g = build_test_graph()
  assert GraphOptimizer([]).optimize(g) is g

  unmatched = GraphOptimizer(
    [PatternDef(name="LinearReLU", sequence=["Linear", "ReLU"], replace_with="LinearReLU", description="test")]
  ).optimize(g)
  assert len(unmatched.nodes) == 5
  assert len(unmatched.edges) == 4


def test_graph_optimizer_single_and_isolated_fusion() -> None:
  """Verify `GraphOptimizer` fuses matched chains, merges attributes, and rewires surrounding edges."""
  g = build_test_graph()
  patterns = [
    PatternDef(name="CBR", sequence=["Conv2d", "BatchNorm", "ReLU"], replace_with="Conv2dBNReLU", description="test")
  ]
  res = GraphOptimizer(patterns).optimize(g)

  assert set(res.nodes.keys()) == {"in", "fused_conv", "out"}
  assert res.nodes["fused_conv"].op_type == "Conv2dBNReLU"
  assert res.nodes["fused_conv"].attributes["arg_0"] == "3"
  assert res.nodes["fused_conv"].attributes["eps"] == "1e-5"
  assert [(e.source, e.target) for e in res.edges] == [("in", "fused_conv"), ("fused_conv", "out")]

  # Isolated 3-node fusion with zero external edges
  isolated = LogicalGraph(
    nodes={
      "1": LogicalNode(id="1", op_type="Conv"),
      "2": LogicalNode(id="2", op_type="BatchNorm"),
      "3": LogicalNode(id="3", op_type="ReLU"),
    },
    edges=[LogicalEdge("1", "2"), LogicalEdge("2", "3")],
  )
  isolated_res = GraphOptimizer(
    [PatternDef(name="CBR", sequence=["Conv", "BatchNorm", "ReLU"], replace_with="FusedConv")]
  ).optimize(isolated)
  assert list(isolated_res.nodes.keys()) == ["fused_1"]
  assert isolated_res.edges == []


def test_graph_optimizer_partial_chain_and_double_fusion() -> None:
  """Verify selective fusion alongside unfused layers and consecutive fused subgraphs."""
  g = LogicalGraph(
    nodes={
      "in": LogicalNode("in", op_type="Input"),
      "l1": LogicalNode("l1", op_type="Linear"),
      "r1": LogicalNode("r1", op_type="ReLU"),
      "l2": LogicalNode("l2", op_type="Linear"),
      "out": LogicalNode("out", op_type="Output"),
    },
    edges=[
      LogicalEdge("in", "l1"),
      LogicalEdge("l1", "r1"),
      LogicalEdge("r1", "l2"),
      LogicalEdge("l2", "out"),
    ],
  )
  res = GraphOptimizer(
    [PatternDef(name="LinearReLU", sequence=["Linear", "ReLU"], replace_with="LinearReLU", description="test")]
  ).optimize(g)
  assert {"in", "fused_l1", "l2", "out"} == set(res.nodes.keys())
  edges = [(e.source, e.target) for e in res.edges]
  assert edges == [("in", "fused_l1"), ("fused_l1", "l2"), ("l2", "out")]

  # Consecutive fused blocks with internal cross-edge dropped
  double_g = LogicalGraph(
    nodes={
      "A1": LogicalNode("A1", op_type="A"),
      "B1": LogicalNode("B1", op_type="B"),
      "A2": LogicalNode("A2", op_type="A"),
      "B2": LogicalNode("B2", op_type="B"),
    },
    edges=[
      LogicalEdge("A1", "B1"),
      LogicalEdge("B1", "A2"),
      LogicalEdge("A2", "B2"),
      LogicalEdge("A1", "B2"),
    ],
  )
  double_res = GraphOptimizer([PatternDef(name="AB", sequence=["A", "B"], replace_with="AB")]).optimize(double_g)
  assert [(e.source, e.target) for e in double_res.edges] == [("fused_A1", "fused_A2")]


def test_optimizer_internal_branch_edges_dropped_and_match_failures() -> None:
  """Verify edges entering non-head or exiting non-tail internal nodes are dropped and `_match_sequence` guards."""
  branch_g = LogicalGraph(
    nodes={
      "0": LogicalNode("0", op_type="Input"),
      "1": LogicalNode("1", op_type="Conv"),
      "2": LogicalNode("2", op_type="BatchNorm"),
      "3": LogicalNode("3", op_type="ReLU"),
      "4": LogicalNode("4", op_type="Output"),
    },
    edges=[
      LogicalEdge("1", "2"),
      LogicalEdge("2", "3"),
      LogicalEdge("1", "4"),
      LogicalEdge("0", "2"),
    ],
  )
  branch_res = GraphOptimizer(
    [PatternDef(name="CBR", sequence=["Conv", "BatchNorm", "ReLU"], replace_with="FusedConv")]
  ).optimize(branch_g)
  assert branch_res.edges == []

  opt = GraphOptimizer([])
  x = LogicalNode("x", op_type="X")
  y = LogicalNode("y", op_type="Y")
  z = LogicalNode("z", op_type="Z")
  assert opt._match_sequence(x, [], {}, {}, set()) is None
  assert opt._match_sequence(x, ["Y", "Z"], {}, {}, set()) is None
  assert opt._match_sequence(x, ["X", "Y"], {"x": x, "y": y}, {"x": []}, set()) is None
  assert opt._match_sequence(x, ["X", "Y"], {"x": x, "y": y}, {"x": ["y"]}, {"y"}) is None
  assert opt._match_sequence(x, ["X", "Y"], {"x": x, "y": z}, {"x": ["y"]}, set()) is None
  assert opt._match_sequence(x, ["X", "Y"], {"x": x, "y": y}, {"x": ["y"]}, set()) == ["x", "y"]
