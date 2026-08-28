"""Test suite for the compiler fusion passes."""

from typing import List, Set, Tuple
from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.core.compiler.fusion import QKVFusionPass, QKVDefusionPass


def test_qkv_fusion_pass_no_nodes() -> None:
  """Test QKVFusionPass with no nodes."""
  graph: LogicalGraph = LogicalGraph(nodes=[], edges=[])
  pass_: QKVFusionPass = QKVFusionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(new_graph.nodes) == 0


def test_qkv_fusion_pass_missing_k_v() -> None:
  """Test QKVFusionPass with missing K and V nodes."""
  graph: LogicalGraph = LogicalGraph(nodes=[LogicalNode(id="q_proj", kind="Linear")], edges=[])
  pass_: QKVFusionPass = QKVFusionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(new_graph.nodes) == 1
  assert new_graph.nodes[0].id == "q_proj"


def test_qkv_fusion_pass_success() -> None:
  """Test QKVFusionPass successfully fusing nodes."""
  nodes: List[LogicalNode] = [
    LogicalNode(id="input", kind="Input"),
    LogicalNode(id="q_proj", kind="Linear"),
    LogicalNode(id="k_proj", kind="Linear"),
    LogicalNode(id="v_proj", kind="Linear"),
    LogicalNode(id="output_q", kind="Output"),
    LogicalNode(id="output_k", kind="Output"),
    LogicalNode(id="output_v", kind="Output"),
  ]
  edges: List[LogicalEdge] = [
    LogicalEdge(source="input", target="q_proj"),
    LogicalEdge(source="input", target="k_proj"),
    LogicalEdge(source="input", target="v_proj"),
    LogicalEdge(source="q_proj", target="output_q"),
    LogicalEdge(source="k_proj", target="output_k"),
    LogicalEdge(source="v_proj", target="output_v"),
  ]
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=edges)
  pass_: QKVFusionPass = QKVFusionPass()
  new_graph: LogicalGraph = pass_.apply(graph)

  node_ids: Set[str] = {n.id for n in new_graph.nodes}
  assert "qkv_proj" in node_ids
  assert "q_proj" not in node_ids
  assert "k_proj" not in node_ids
  assert "v_proj" not in node_ids

  # Check edges
  edge_sources: Set[Tuple[str, str]] = {(e.source, e.target) for e in new_graph.edges}
  assert ("input", "qkv_proj") in edge_sources
  assert ("qkv_proj", "output_q") in edge_sources
  assert ("qkv_proj", "output_k") in edge_sources
  assert ("qkv_proj", "output_v") in edge_sources


def test_qkv_defusion_pass_no_nodes() -> None:
  """Test QKVDefusionPass with no nodes."""
  graph: LogicalGraph = LogicalGraph(nodes=[], edges=[])
  pass_: QKVDefusionPass = QKVDefusionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(new_graph.nodes) == 0


def test_qkv_defusion_pass_success() -> None:
  """Test QKVDefusionPass successfully de-fusing nodes."""
  nodes: List[LogicalNode] = [
    LogicalNode(id="input", kind="Input"),
    LogicalNode(id="qkv_proj", kind="Linear"),
    LogicalNode(id="output", kind="Output"),
  ]
  edges: List[LogicalEdge] = [
    LogicalEdge(source="input", target="qkv_proj"),
    LogicalEdge(source="qkv_proj", target="output"),
  ]
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=edges)
  pass_: QKVDefusionPass = QKVDefusionPass()
  new_graph: LogicalGraph = pass_.apply(graph)

  node_ids: Set[str] = {n.id for n in new_graph.nodes}
  assert "qkv_proj" not in node_ids
  assert "q_proj" in node_ids
  assert "k_proj" in node_ids
  assert "v_proj" in node_ids

  # Check edges
  edge_sources: Set[Tuple[str, str]] = {(e.source, e.target) for e in new_graph.edges}
  assert ("input", "q_proj") in edge_sources
  assert ("input", "k_proj") in edge_sources
  assert ("input", "v_proj") in edge_sources
  assert ("q_proj", "output") in edge_sources
  assert ("k_proj", "output") in edge_sources
  assert ("v_proj", "output") in edge_sources


def test_qkv_fusion_pass_unrelated_edge() -> None:
  """Test QKVFusionPass with unrelated edge."""
  nodes: List[LogicalNode] = [
    LogicalNode(id="input", kind="Input"),
    LogicalNode(id="q_proj", kind="Linear"),
    LogicalNode(id="k_proj", kind="Linear"),
    LogicalNode(id="v_proj", kind="Linear"),
    LogicalNode(id="output_q", kind="Output"),
    LogicalNode(id="unrelated1", kind="Other"),
    LogicalNode(id="unrelated2", kind="Other"),
  ]
  edges: List[LogicalEdge] = [
    LogicalEdge(source="input", target="q_proj"),
    LogicalEdge(source="input", target="k_proj"),
    LogicalEdge(source="input", target="v_proj"),
    LogicalEdge(source="q_proj", target="output_q"),
    LogicalEdge(source="unrelated1", target="unrelated2"),  # unrelated
  ]
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=edges)
  pass_: QKVFusionPass = QKVFusionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  edge_sources: Set[Tuple[str, str]] = {(e.source, e.target) for e in new_graph.edges}
  assert ("unrelated1", "unrelated2") in edge_sources


def test_qkv_defusion_pass_unrelated_edge() -> None:
  """Test QKVDefusionPass with unrelated edge."""
  nodes: List[LogicalNode] = [
    LogicalNode(id="input", kind="Input"),
    LogicalNode(id="qkv_proj", kind="Linear"),
    LogicalNode(id="output", kind="Output"),
    LogicalNode(id="unrelated1", kind="Other"),
    LogicalNode(id="unrelated2", kind="Other"),
  ]
  edges: List[LogicalEdge] = [
    LogicalEdge(source="input", target="qkv_proj"),
    LogicalEdge(source="qkv_proj", target="output"),
    LogicalEdge(source="unrelated1", target="unrelated2"),  # unrelated
  ]
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=edges)
  pass_: QKVDefusionPass = QKVDefusionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  edge_sources: Set[Tuple[str, str]] = {(e.source, e.target) for e in new_graph.edges}
  assert ("unrelated1", "unrelated2") in edge_sources
