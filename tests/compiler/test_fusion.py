"""Test suite for the Fusion module."""

from ml_switcheroo.core.compiler.fusion import QKVDefusionPass, QKVFusionPass
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode


def test_qkv_fusion_pass() -> None:
  """Verifies the behavior of qkv fusion pass."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="input", kind="Input"),
      LogicalNode(id="q_proj", kind="Linear"),
      LogicalNode(id="k_proj", kind="Linear"),
      LogicalNode(id="v_proj", kind="Linear"),
      LogicalNode(id="attention", kind="Attention"),
      LogicalNode(id="other", kind="Other"),
    ],
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
  node_ids: set[str] = {n.id for n in fused_graph.nodes}
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
  graph = LogicalGraph(nodes=[LogicalNode(id="q_proj", kind="Linear")])
  pass_ = QKVFusionPass()
  fused_graph: LogicalGraph = pass_.apply(graph)
  assert len(fused_graph.nodes) == 1


def test_qkv_defusion_pass() -> None:
  """Verifies the behavior of qkv defusion pass."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="input", kind="Input"),
      LogicalNode(id="qkv_proj", kind="Linear"),
      LogicalNode(id="attention", kind="Attention"),
      LogicalNode(id="other", kind="Other"),
    ],
    edges=[LogicalEdge("input", "qkv_proj"), LogicalEdge("qkv_proj", "attention"), LogicalEdge("input", "other")],
  )
  pass_ = QKVDefusionPass()
  defused_graph: LogicalGraph = pass_.apply(graph)
  node_ids: set[str] = {n.id for n in defused_graph.nodes}
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
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="layer1_q_proj", kind="Linear"),
      LogicalNode(id="layer2_k_proj", kind="Linear"),
      LogicalNode(id="layer3_v_proj", kind="Linear"),
    ],
    edges=[],
  )
  pass_ = QKVFusionPass()
  fused_graph: LogicalGraph = pass_.apply(graph)
  # No fusion should occur
  assert len(fused_graph.nodes) == 3


def test_qkv_defusion_pass_no_match() -> None:
  """Verifies the behavior when qkv_proj doesn't exist."""
  graph = LogicalGraph(nodes=[LogicalNode(id="q_proj", kind="Linear")])
  pass_ = QKVDefusionPass()
  defused: LogicalGraph = pass_.apply(graph)
  assert len(defused.nodes) == 1
