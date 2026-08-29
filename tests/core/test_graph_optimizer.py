"""Test module."""

import typing

from ml_switcheroo.core.dsl import PatternDef
from ml_switcheroo.core.graph import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.graph_optimizer import GraphOptimizer


def build_test_graph() -> LogicalGraph:
  """Docstring."""
  # Input -> Conv2d -> BatchNorm -> ReLU -> Output
  g = LogicalGraph()
  g.nodes = [
    LogicalNode("in", "Input"),
    LogicalNode("conv", "Conv2d", {"arg_0": "3", "arg_1": "16"}),
    LogicalNode("bn", "BatchNorm", {"eps": "1e-5"}),
    LogicalNode("relu", "ReLU"),
    LogicalNode("out", "Output"),
  ]
  g.edges = [LogicalEdge("in", "conv"), LogicalEdge("conv", "bn"), LogicalEdge("bn", "relu"), LogicalEdge("relu", "out")]
  return g


def test_graph_optimizer_no_patterns() -> None:
  """Docstring."""
  g: LogicalGraph = build_test_graph()
  opt = GraphOptimizer([])
  res: LogicalGraph = opt.optimize(g)
  # The original graph object is returned directly
  assert res is g


def test_graph_optimizer_single_pattern() -> None:
  """Docstring."""
  g: LogicalGraph = build_test_graph()
  patterns = [
    PatternDef(name="CBR", sequence=["Conv2d", "BatchNorm", "ReLU"], replace_with="Conv2dBNReLU", description="test")
  ]
  opt = GraphOptimizer(patterns)
  res: LogicalGraph = opt.optimize(g)

  node_ids: dict[str, LogicalNode] = {n.id: n for n in res.nodes}
  assert "fused_conv" in node_ids
  assert node_ids["fused_conv"].kind == "Conv2dBNReLU"
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
  fused_meta: dict[str, typing.Any] = node_ids["fused_conv"].metadata
  assert fused_meta["arg_0"] == "3"
  assert fused_meta["eps"] == "1e-5"


def test_graph_optimizer_no_match() -> None:
  """Docstring."""
  g: LogicalGraph = build_test_graph()
  patterns = [PatternDef(name="LinearReLU", sequence=["Linear", "ReLU"], replace_with="LinearReLU", description="test")]
  opt = GraphOptimizer(patterns)
  res: LogicalGraph = opt.optimize(g)

  assert len(res.nodes) == 5
  assert len(res.edges) == 4


def test_graph_optimizer_multiple_patterns() -> None:
  """Docstring."""
  # Input -> Linear -> ReLU -> Linear -> Output
  g = LogicalGraph()
  g.nodes = [
    LogicalNode("in", "Input"),
    LogicalNode("l1", "Linear"),
    LogicalNode("r1", "ReLU"),
    LogicalNode("l2", "Linear"),
    LogicalNode("out", "Output"),
  ]
  g.edges = [LogicalEdge("in", "l1"), LogicalEdge("l1", "r1"), LogicalEdge("r1", "l2"), LogicalEdge("l2", "out")]

  patterns = [PatternDef(name="LinearReLU", sequence=["Linear", "ReLU"], replace_with="LinearReLU", description="test")]
  opt = GraphOptimizer(patterns)
  res: LogicalGraph = opt.optimize(g)

  node_ids: dict[str, LogicalNode] = {n.id: n for n in res.nodes}
  assert "fused_l1" in node_ids
  assert "l2" in node_ids

  edges: list[tuple[str, str]] = [(e.source, e.target) for e in res.edges]
  assert ("in", "fused_l1") in edges
  assert ("fused_l1", "l2") in edges
  assert ("l2", "out") in edges


def test_match_sequence_fail_empty_seq() -> None:
  """Docstring."""
  opt = GraphOptimizer([])
  n = LogicalNode("x", "X")
  assert opt._match_sequence(n, [], {}, {}, set()) is None


def test_match_sequence_fail_first_node() -> None:
  """Docstring."""
  opt = GraphOptimizer([])
  n = LogicalNode("x", "X")
  assert opt._match_sequence(n, ["Y", "Z"], {}, {}, set()) is None


def test_match_sequence_fail_missing_target() -> None:
  """Docstring."""
  opt = GraphOptimizer([])
  n = LogicalNode("x", "X")
  node_map: dict[str, LogicalNode] = {"x": n, "y": LogicalNode("y", "Y")}
  # No out edge from x to y
  out_edges: dict[str, list[str]] = {"x": []}
  assert opt._match_sequence(n, ["X", "Y"], node_map, out_edges, set()) is None


def test_match_sequence_fail_already_processed() -> None:
  """Docstring."""
  opt = GraphOptimizer([])
  n = LogicalNode("x", "X")
  y = LogicalNode("y", "Y")
  node_map: dict[str, LogicalNode] = {"x": n, "y": y}
  out_edges: dict[str, list[str]] = {"x": ["y"]}
  assert opt._match_sequence(n, ["X", "Y"], node_map, out_edges, {"y"}) is None


def test_match_sequence_fail_wrong_kind() -> None:
  """Docstring."""
  opt = GraphOptimizer([])
  n = LogicalNode("x", "X")
  y = LogicalNode("y", "Z")  # Wrong kind
  node_map: dict[str, LogicalNode] = {"x": n, "y": y}
  out_edges: dict[str, list[str]] = {"x": ["y"]}
  assert opt._match_sequence(n, ["X", "Y"], node_map, out_edges, set()) is None


def test_optimizer_branching_edge_drops() -> None:
  """Docstring."""
  # Check that edges internal to fusion block drop
  # And check cross-fusion links drop if internal
  g = LogicalGraph()
  g.nodes = [LogicalNode("A", "OpA"), LogicalNode("B", "OpB"), LogicalNode("C", "OpC")]
  # A -> B -> C
  # A -> C (bypass edge, originates from internal part)
  g.edges = [LogicalEdge("A", "B"), LogicalEdge("B", "C"), LogicalEdge("A", "C")]

  p = PatternDef(name="AB", sequence=["OpA", "OpB"], replace_with="OpAB", description="")
  opt = GraphOptimizer([p])

  res: LogicalGraph = opt.optimize(g)

  edges: list[tuple[str, str]] = [(e.source, e.target) for e in res.edges]
  # fused_A represents A+B. A was head, B was tail.
  # The A->B edge is internal.
  # The B->C edge is (fused_A.tail) -> C, so it becomes fused_A -> C
  # The A->C edge is (fused_A.head) -> C, which is internal non-tail, so it drops

  assert ("fused_A", "C") in edges
  assert ("A", "C") not in edges
  assert ("fused_A", "B") not in edges
  assert len(edges) == 1


def test_optimizer_double_fusion_link() -> None:
  """Docstring."""
  g = LogicalGraph()
  g.nodes = [LogicalNode("A1", "A"), LogicalNode("B1", "B"), LogicalNode("A2", "A"), LogicalNode("B2", "B")]
  g.edges = [
    LogicalEdge("A1", "B1"),
    LogicalEdge("B1", "A2"),
    LogicalEdge("A2", "B2"),
    LogicalEdge("A1", "B2"),  # A1 to B2 is internal to both, should drop
  ]
  p = PatternDef(name="AB", sequence=["A", "B"], replace_with="AB", description="")
  opt = GraphOptimizer([p])
  res: LogicalGraph = opt.optimize(g)

  edges: list[tuple[str, str]] = [(e.source, e.target) for e in res.edges]
  # B1 (tail of fused_A1) -> A2 (head of fused_A2) -> fused_A1 -> fused_A2
  assert ("fused_A1", "fused_A2") in edges
  assert len(edges) == 1
