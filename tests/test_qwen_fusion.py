"""Docstring."""

from typing import Dict
from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.compiler.qwen_fusion import (
  SwiGLUDefusionPass,
  SwiGLUFusionPass,
  VisionPatchEmbeddingDefusionPass,
  VisionPatchEmbeddingFusionPass,
)


def test_swiglu_fusion_pass() -> None:
  """Docstring."""
  nodes: Dict[str, LogicalNode] = {
    "input": LogicalNode(id="input", op_type="Input"),
    "layer_gate_proj": LogicalNode(id="layer_gate_proj", op_type="Linear"),
    "layer_up_proj": LogicalNode(id="layer_up_proj", op_type="Linear"),
    "layer_out": LogicalNode(id="layer_out", op_type="Other"),
    "unrelated1": LogicalNode(id="unrelated1", op_type="Other"),
    "unrelated2": LogicalNode(id="unrelated2", op_type="Other"),
  }
  graph: LogicalGraph = LogicalGraph(
    nodes=nodes,
    edges=[
      LogicalEdge(source="input", target="layer_gate_proj"),
      LogicalEdge(source="input", target="layer_up_proj"),
      LogicalEdge(source="layer_gate_proj", target="layer_out"),
      LogicalEdge(source="layer_up_proj", target="layer_out"),
      LogicalEdge(source="unrelated1", target="unrelated2"),
    ],
  )
  pass_: SwiGLUFusionPass = SwiGLUFusionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert any(n.id == "layer_swiglu" for n in new_graph.nodes.values())
  assert any(e.source == "unrelated1" for e in new_graph.edges)


def test_swiglu_fusion_pass_no_match() -> None:
  """Docstring."""
  nodes: Dict[str, LogicalNode] = {"input": LogicalNode(id="input", op_type="Input")}
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=[])
  pass_: SwiGLUFusionPass = SwiGLUFusionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(new_graph.nodes) == 1


def test_swiglu_defusion_pass() -> None:
  """Docstring."""
  nodes: Dict[str, LogicalNode] = {
    "input": LogicalNode(id="input", op_type="Input"),
    "layer_swiglu": LogicalNode(id="layer_swiglu", op_type="SwiGLU"),
    "layer_out": LogicalNode(id="layer_out", op_type="Other"),
    "unrelated1": LogicalNode(id="unrelated1", op_type="Other"),
    "unrelated2": LogicalNode(id="unrelated2", op_type="Other"),
  }
  graph: LogicalGraph = LogicalGraph(
    nodes=nodes,
    edges=[
      LogicalEdge(source="input", target="layer_swiglu"),
      LogicalEdge(source="layer_swiglu", target="layer_out"),
      LogicalEdge(source="unrelated1", target="unrelated2"),
    ],
  )
  pass_: SwiGLUDefusionPass = SwiGLUDefusionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert any(n.id == "layer_gate_proj" for n in new_graph.nodes.values())
  assert any(n.id == "layer_up_proj" for n in new_graph.nodes.values())
  assert any(e.source == "unrelated1" for e in new_graph.edges)


def test_swiglu_defusion_pass_no_match() -> None:
  """Docstring."""
  nodes: Dict[str, LogicalNode] = {"input": LogicalNode(id="input", op_type="Input")}
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=[])
  pass_: SwiGLUDefusionPass = SwiGLUDefusionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(new_graph.nodes) == 1


def test_vision_patch_fusion_pass() -> None:
  """Docstring."""
  nodes: Dict[str, LogicalNode] = {
    "input": LogicalNode(id="input", op_type="Input"),
    "patch_conv": LogicalNode(id="patch_conv", op_type="Conv2d"),
    "output": LogicalNode(id="output", op_type="Output"),
    "unrelated1": LogicalNode(id="unrelated1", op_type="Other"),
    "unrelated2": LogicalNode(id="unrelated2", op_type="Other"),
  }
  graph: LogicalGraph = LogicalGraph(
    nodes=nodes,
    edges=[
      LogicalEdge(source="input", target="patch_conv"),
      LogicalEdge(source="patch_conv", target="output"),
      LogicalEdge(source="unrelated1", target="unrelated2"),
    ],
  )
  pass_: VisionPatchEmbeddingFusionPass = VisionPatchEmbeddingFusionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert any(
    (getattr(n, "op_type", None) or getattr(n, "kind", "")) == "VisionPatchEmbedding" for n in new_graph.nodes.values()
  )
  assert any(e.source == "unrelated1" for e in new_graph.edges)


def test_vision_patch_fusion_pass_no_match() -> None:
  """Docstring."""
  nodes: Dict[str, LogicalNode] = {"input": LogicalNode(id="input", op_type="Input")}
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=[])
  pass_: VisionPatchEmbeddingFusionPass = VisionPatchEmbeddingFusionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(new_graph.nodes) == 1


def test_vision_patch_defusion_pass() -> None:
  """Docstring."""
  nodes: Dict[str, LogicalNode] = {
    "input": LogicalNode(id="input", op_type="Input"),
    "patch_embed": LogicalNode(id="patch_embed", op_type="VisionPatchEmbedding"),
    "output": LogicalNode(id="output", op_type="Output"),
    "unrelated1": LogicalNode(id="unrelated1", op_type="Other"),
    "unrelated2": LogicalNode(id="unrelated2", op_type="Other"),
  }
  graph: LogicalGraph = LogicalGraph(
    nodes=nodes,
    edges=[
      LogicalEdge(source="input", target="patch_embed"),
      LogicalEdge(source="patch_embed", target="output"),
      LogicalEdge(source="unrelated1", target="unrelated2"),
    ],
  )
  pass_: VisionPatchEmbeddingDefusionPass = VisionPatchEmbeddingDefusionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert any((getattr(n, "op_type", None) or getattr(n, "kind", "")) == "Conv2d" for n in new_graph.nodes.values())
  assert any(e.source == "unrelated1" for e in new_graph.edges)


def test_vision_patch_defusion_pass_no_match() -> None:
  """Docstring."""
  nodes: Dict[str, LogicalNode] = {"input": LogicalNode(id="input", op_type="Input")}
  graph: LogicalGraph = LogicalGraph(nodes=nodes, edges=[])
  pass_: VisionPatchEmbeddingDefusionPass = VisionPatchEmbeddingDefusionPass()
  new_graph: LogicalGraph = pass_.apply(graph)
  assert len(new_graph.nodes) == 1
