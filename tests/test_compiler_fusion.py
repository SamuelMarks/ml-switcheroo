"""Test suite for the compiler fusion passes."""

from typing import Dict, Set, Tuple

from ml_switcheroo.core.compiler.fusion import QKVDefusionPass, QKVFusionPass
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode


def test_qkv_fusion_pass_no_nodes() -> None:
  """Docstring."""
  graph: LogicalGraph = LogicalGraph(nodes={}, edges=[])
  pass_: QKVFusionPass = QKVFusionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(new_graph.nodes) == 0


def test_qkv_fusion_pass_missing_k_v() -> None:
  """Docstring."""
  nodes: Dict[str, LogicalNode] = {"q_proj": LogicalNode(id="q_proj", op_type="Linear")}
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=[])
  pass_: QKVFusionPass = QKVFusionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(new_graph.nodes) == 1
  assert "q_proj" in new_graph.nodes
  assert new_graph.nodes["q_proj"].id == "q_proj"


def test_qkv_fusion_pass_success() -> None:
  """Docstring."""
  nodes: Dict[str, LogicalNode] = {
    "input": LogicalNode(id="input", op_type="Input"),
    "q_proj": LogicalNode(id="q_proj", op_type="Linear"),
    "k_proj": LogicalNode(id="k_proj", op_type="Linear"),
    "v_proj": LogicalNode(id="v_proj", op_type="Linear"),
    "output_q": LogicalNode(id="output_q", op_type="Output"),
    "output_k": LogicalNode(id="output_k", op_type="Output"),
    "output_v": LogicalNode(id="output_v", op_type="Output"),
  }
  edges = [
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

  node_ids: Set[str] = set(new_graph.nodes.keys())
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
  """Docstring."""
  graph: LogicalGraph = LogicalGraph(nodes={}, edges=[])
  pass_: QKVDefusionPass = QKVDefusionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(new_graph.nodes) == 0


def test_qkv_defusion_pass_success() -> None:
  """Docstring."""
  nodes: Dict[str, LogicalNode] = {
    "input": LogicalNode(id="input", op_type="Input"),
    "qkv_proj": LogicalNode(id="qkv_proj", op_type="Linear"),
    "output": LogicalNode(id="output", op_type="Output"),
  }
  edges = [
    LogicalEdge(source="input", target="qkv_proj"),
    LogicalEdge(source="qkv_proj", target="output"),
  ]
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=edges)
  pass_: QKVDefusionPass = QKVDefusionPass()
  new_graph: LogicalGraph = pass_.apply(graph)

  node_ids: Set[str] = set(new_graph.nodes.keys())
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
  """Docstring."""
  nodes: Dict[str, LogicalNode] = {
    "input": LogicalNode(id="input", op_type="Input"),
    "q_proj": LogicalNode(id="q_proj", op_type="Linear"),
    "k_proj": LogicalNode(id="k_proj", op_type="Linear"),
    "v_proj": LogicalNode(id="v_proj", op_type="Linear"),
    "output_q": LogicalNode(id="output_q", op_type="Output"),
    "unrelated1": LogicalNode(id="unrelated1", op_type="Other"),
    "unrelated2": LogicalNode(id="unrelated2", op_type="Other"),
  }
  edges = [
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
  """Docstring."""
  nodes: Dict[str, LogicalNode] = {
    "input": LogicalNode(id="input", op_type="Input"),
    "qkv_proj": LogicalNode(id="qkv_proj", op_type="Linear"),
    "output": LogicalNode(id="output", op_type="Output"),
    "unrelated1": LogicalNode(id="unrelated1", op_type="Other"),
    "unrelated2": LogicalNode(id="unrelated2", op_type="Other"),
  }
  edges = [
    LogicalEdge(source="input", target="qkv_proj"),
    LogicalEdge(source="qkv_proj", target="output"),
    LogicalEdge(source="unrelated1", target="unrelated2"),  # unrelated
  ]
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=edges)
  pass_: QKVDefusionPass = QKVDefusionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  edge_sources: Set[Tuple[str, str]] = {(e.source, e.target) for e in new_graph.edges}
  assert ("unrelated1", "unrelated2") in edge_sources
