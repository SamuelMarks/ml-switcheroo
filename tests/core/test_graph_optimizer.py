"""Test suite for graph optimizer."""

import typing

from ml_switcheroo.core.dsl import PatternDef
from ml_switcheroo.core.graph import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.graph_optimizer import GraphOptimizer


def build_test_graph() -> LogicalGraph:
  """Execute build test graph helper logic.

  Returns:
      Computed test result or fixture value.
  """
  # Input -> Conv2d -> BatchNorm -> ReLU -> Output
  nodes: typing.Any = {
    "in": LogicalNode("in", op_type="Input"),
    "conv": LogicalNode("conv", op_type="Conv2d", attributes={"arg_0": "3", "arg_1": "16"}),
    "bn": LogicalNode("bn", op_type="BatchNorm", attributes={"eps": "1e-5"}),
    "relu": LogicalNode("relu", op_type="ReLU"),
    "out": LogicalNode("out", op_type="Output"),
  }
  edges: typing.Any = [
    LogicalEdge("in", "conv"),
    LogicalEdge("conv", "bn"),
    LogicalEdge("bn", "relu"),
    LogicalEdge("relu", "out"),
  ]
  return LogicalGraph(nodes=nodes, edges=edges)


def test_graph_optimizer_no_patterns() -> None:
  """Verify graph optimizer no patterns behavior."""
  g: LogicalGraph = build_test_graph()
  opt: typing.Any = GraphOptimizer([])
  res: LogicalGraph = opt.optimize(g)
  # The original graph object is returned directly
  assert res is g


def test_graph_optimizer_single_pattern() -> None:
  """Verify graph optimizer single pattern behavior."""
  g: LogicalGraph = build_test_graph()
  patterns: typing.Any = [
    PatternDef(name="CBR", sequence=["Conv2d", "BatchNorm", "ReLU"], replace_with="Conv2dBNReLU", description="test")
  ]
  opt: typing.Any = GraphOptimizer(patterns)
  res: LogicalGraph = opt.optimize(g)

  node_ids: dict[str, LogicalNode] = dict(res.nodes)
  assert "fused_conv" in node_ids
  assert node_ids["fused_conv"].op_type == "Conv2dBNReLU"
  assert "conv" not in node_ids
  assert "bn" not in node_ids
  assert "relu" not in node_ids
  assert "in" in node_ids
  assert "out" in node_ids

  edges: list[tuple[str, str]] = [(e.source, e.target) for e in res.edges]
  assert ("in", "fused_conv") in edges
  assert ("fused_conv", "out") in edges
  assert len(edges) == 2

  # Metadata merging
  fused_meta: dict[str, typing.Any] = node_ids["fused_conv"].attributes
  assert fused_meta["arg_0"] == "3"
  assert fused_meta["eps"] == "1e-5"


def test_graph_optimizer_no_match() -> None:
  """Verify graph optimizer no match behavior."""
  g: LogicalGraph = build_test_graph()
  patterns: typing.Any = [
    PatternDef(name="LinearReLU", sequence=["Linear", "ReLU"], replace_with="LinearReLU", description="test")
  ]
  opt: typing.Any = GraphOptimizer(patterns)
  res: LogicalGraph = opt.optimize(g)

  assert len(res.nodes) == 5
  assert len(res.edges) == 4


def test_graph_optimizer_multiple_patterns() -> None:
  """Verify graph optimizer multiple patterns behavior."""
  # Input -> Linear -> ReLU -> Linear -> Output
  nodes: typing.Any = {
    "in": LogicalNode("in", op_type="Input"),
    "l1": LogicalNode("l1", op_type="Linear"),
    "r1": LogicalNode("r1", op_type="ReLU"),
    "l2": LogicalNode("l2", op_type="Linear"),
    "out": LogicalNode("out", op_type="Output"),
  }
  edges: typing.Any = [
    LogicalEdge("in", "l1"),
    LogicalEdge("l1", "r1"),
    LogicalEdge("r1", "l2"),
    LogicalEdge("l2", "out"),
  ]
  g: typing.Any = LogicalGraph(nodes=nodes, edges=edges)

  patterns: typing.Any = [
    PatternDef(name="LinearReLU", sequence=["Linear", "ReLU"], replace_with="LinearReLU", description="test")
  ]
  opt: typing.Any = GraphOptimizer(patterns)
  res: LogicalGraph = opt.optimize(g)

  node_ids: dict[str, LogicalNode] = dict(res.nodes)
  assert "fused_l1" in node_ids
  assert "l2" in node_ids

  edges = [(e.source, e.target) for e in res.edges]
  assert ("in", "fused_l1") in edges
  assert ("fused_l1", "l2") in edges
  assert ("l2", "out") in edges


def test_match_sequence_fail_empty_seq() -> None:
  """Verify match sequence fail empty seq behavior."""
  opt: typing.Any = GraphOptimizer([])
  n: typing.Any = LogicalNode("x", op_type="X")
  assert opt._match_sequence(n, [], {}, {}, set()) is None


def test_match_sequence_fail_first_node() -> None:
  """Verify match sequence fail first node behavior."""
  opt: typing.Any = GraphOptimizer([])
  n: typing.Any = LogicalNode("x", op_type="X")
  assert opt._match_sequence(n, ["Y", "Z"], {}, {}, set()) is None


def test_match_sequence_fail_missing_target() -> None:
  """Verify match sequence fail missing target behavior."""
  opt: typing.Any = GraphOptimizer([])
  n: typing.Any = LogicalNode("x", op_type="X")
  node_map: dict[str, LogicalNode] = {"x": n, "y": LogicalNode("y", op_type="Y")}
  # No out edge from x to y
  out_edges: dict[str, list[str]] = {"x": []}
  assert opt._match_sequence(n, ["X", "Y"], node_map, out_edges, set()) is None


def test_match_sequence_fail_already_processed() -> None:
  """Verify match sequence fail already processed behavior."""
  opt: typing.Any = GraphOptimizer([])
  n: typing.Any = LogicalNode("x", op_type="X")
  y: typing.Any = LogicalNode("y", op_type="Y")
  node_map: dict[str, LogicalNode] = {"x": n, "y": y}
  out_edges: dict[str, list[str]] = {"x": ["y"]}
  assert opt._match_sequence(n, ["X", "Y"], node_map, out_edges, {"y"}) is None


def test_match_sequence_fail_wrong_kind() -> None:
  """Verify match sequence fail wrong kind behavior."""
  opt: typing.Any = GraphOptimizer([])
  n: typing.Any = LogicalNode("x", op_type="X")
  y: typing.Any = LogicalNode("y", op_type="Z")  # Wrong kind
  node_map: dict[str, LogicalNode] = {"x": n, "y": y}
  out_edges: dict[str, list[str]] = {"x": ["y"]}
  assert opt._match_sequence(n, ["X", "Y"], node_map, out_edges, set()) is None


def test_optimizer_branching_edge_drops() -> None:
  """Verify optimizer branching edge drops behavior."""
  nodes: typing.Any = {
    "A": LogicalNode("A", op_type="OpA"),
    "B": LogicalNode("B", op_type="OpB"),
    "C": LogicalNode("C", op_type="OpC"),
  }
  edges: typing.Any = [LogicalEdge("A", "B"), LogicalEdge("B", "C"), LogicalEdge("A", "C")]
  g: typing.Any = LogicalGraph(nodes=nodes, edges=edges)

  p: typing.Any = PatternDef(name="AB", sequence=["OpA", "OpB"], replace_with="OpAB", description="")
  opt: typing.Any = GraphOptimizer([p])

  res: LogicalGraph = opt.optimize(g)

  edges_res: list[tuple[str, str]] = [(e.source, e.target) for e in res.edges]
  assert ("fused_A", "C") in edges_res
  assert ("A", "C") not in edges_res
  assert ("fused_A", "B") not in edges_res
  assert len(edges_res) == 1


def test_optimizer_double_fusion_link() -> None:
  """Verify optimizer double fusion link behavior."""
  nodes: typing.Any = {
    "A1": LogicalNode("A1", op_type="A"),
    "B1": LogicalNode("B1", op_type="B"),
    "A2": LogicalNode("A2", op_type="A"),
    "B2": LogicalNode("B2", op_type="B"),
  }
  edges: typing.Any = [
    LogicalEdge("A1", "B1"),
    LogicalEdge("B1", "A2"),
    LogicalEdge("A2", "B2"),
    LogicalEdge("A1", "B2"),  # A1 to B2 is internal to both, should drop
  ]
  g: typing.Any = LogicalGraph(nodes=nodes, edges=edges)
  p: typing.Any = PatternDef(name="AB", sequence=["A", "B"], replace_with="AB", description="")
  opt: typing.Any = GraphOptimizer([p])
  res: LogicalGraph = opt.optimize(g)

  edges_res: list[tuple[str, str]] = [(e.source, e.target) for e in res.edges]
  assert ("fused_A1", "fused_A2") in edges_res
  assert len(edges_res) == 1
