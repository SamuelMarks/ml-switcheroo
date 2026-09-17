"""Test suite for the Fusion module."""

from ml_switcheroo.core.compiler.fusion import QKVDefusionPass, QKVFusionPass
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode


def test_qkv_fusion_pass() -> None:
  """Verifies the behavior of qkv fusion pass."""
  nodes = {
    "input": LogicalNode(id="input", op_type="Input"),
    "q_proj": LogicalNode(id="q_proj", op_type="Linear"),
    "k_proj": LogicalNode(id="k_proj", op_type="Linear"),
    "v_proj": LogicalNode(id="v_proj", op_type="Linear"),
    "attention": LogicalNode(id="attention", op_type="Attention"),
    "other": LogicalNode(id="other", op_type="Other"),
  }
  graph = LogicalGraph(
    nodes=nodes,
    edges=[
      LogicalEdge("input", "q_proj"),
      LogicalEdge("input", "k_proj"),
      LogicalEdge("input", "v_proj"),
      LogicalEdge("q_proj", "attention"),
      LogicalEdge("k_proj", "attention"),
      LogicalEdge("v_proj", "attention"),
      LogicalEdge("input", "other"),
    ],
  )
  pass_ = QKVFusionPass()
  fused_graph: LogicalGraph = pass_.apply(graph)
  node_ids: set[str] = set(fused_graph.nodes.keys())
  assert "qkv_proj" in node_ids
  assert "q_proj" not in node_ids
  assert "k_proj" not in node_ids
  assert "v_proj" not in node_ids
  assert "other" in node_ids
  edges: list[tuple[str, str]] = [(e.source, e.target) for e in fused_graph.edges]
  assert ("input", "qkv_proj") in edges
  assert ("qkv_proj", "attention") in edges
  assert ("input", "other") in edges


def test_qkv_fusion_pass_no_match() -> None:
  """Verifies the behavior of qkv fusion pass no match."""
  graph = LogicalGraph(nodes={"q_proj": LogicalNode(id="q_proj", op_type="Linear")})
  pass_ = QKVFusionPass()
  fused_graph: LogicalGraph = pass_.apply(graph)
  assert len(fused_graph.nodes) == 1


def test_qkv_defusion_pass() -> None:
  """Verifies the behavior of qkv defusion pass."""
  nodes = {
    "input": LogicalNode(id="input", op_type="Input"),
    "qkv_proj": LogicalNode(id="qkv_proj", op_type="Linear"),
    "attention": LogicalNode(id="attention", op_type="Attention"),
    "other": LogicalNode(id="other", op_type="Other"),
  }
  graph = LogicalGraph(
    nodes=nodes,
    edges=[LogicalEdge("input", "qkv_proj"), LogicalEdge("qkv_proj", "attention"), LogicalEdge("input", "other")],
  )
  pass_ = QKVDefusionPass()
  defused_graph: LogicalGraph = pass_.apply(graph)
  node_ids: set[str] = set(defused_graph.nodes.keys())
  assert "qkv_proj" not in node_ids
  assert "q_proj" in node_ids
  assert "k_proj" in node_ids
  assert "v_proj" in node_ids
  assert "other" in node_ids
  edges: list[tuple[str, str]] = [(e.source, e.target) for e in defused_graph.edges]
  assert ("input", "q_proj") in edges
  assert ("q_proj", "attention") in edges
  assert ("input", "other") in edges


def test_qkv_fusion_pass_unmatched_prefix() -> None:
  """Verifies the behavior when q, k, v exist but prefixes don't match."""
  nodes = {
    "layer1_q_proj": LogicalNode(id="layer1_q_proj", op_type="Linear"),
    "layer2_k_proj": LogicalNode(id="layer2_k_proj", op_type="Linear"),
    "layer3_v_proj": LogicalNode(id="layer3_v_proj", op_type="Linear"),
  }
  graph = LogicalGraph(
    nodes=nodes,
    edges=[],
  )
  pass_ = QKVFusionPass()
  fused_graph: LogicalGraph = pass_.apply(graph)
  # No fusion should occur
  assert len(fused_graph.nodes) == 3


def test_qkv_defusion_pass_no_match() -> None:
  """Verifies the behavior when qkv_proj doesn't exist."""
  graph = LogicalGraph(nodes={"q_proj": LogicalNode(id="q_proj", op_type="Linear")})
  pass_ = QKVDefusionPass()
  defused: LogicalGraph = pass_.apply(graph)
  assert len(defused.nodes) == 1
