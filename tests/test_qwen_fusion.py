"""Docstring."""

from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.core.compiler.qwen_fusion import (
  SwiGLUFusionPass,
  SwiGLUDefusionPass,
  VisionPatchEmbeddingFusionPass,
  VisionPatchEmbeddingDefusionPass,
)


def test_swiglu_fusion_pass():
  """Docstring."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="input", kind="Input"),
      LogicalNode(id="layer_gate_proj", kind="Linear"),
      LogicalNode(id="layer_up_proj", kind="Linear"),
      LogicalNode(id="layer_out", kind="Other"),
      LogicalNode(id="unrelated1", kind="Other"),
      LogicalNode(id="unrelated2", kind="Other"),
    ],
    edges=[
      LogicalEdge(source="input", target="layer_gate_proj"),
      LogicalEdge(source="input", target="layer_up_proj"),
      LogicalEdge(source="layer_gate_proj", target="layer_out"),
      LogicalEdge(source="layer_up_proj", target="layer_out"),
      LogicalEdge(source="unrelated1", target="unrelated2"),
    ],
  )
  pass_ = SwiGLUFusionPass()
  new_graph = pass_.apply(graph)
  assert any(n.id == "layer_swiglu" for n in new_graph.nodes)
  assert any(e.source == "unrelated1" for e in new_graph.edges)


def test_swiglu_fusion_pass_no_match():
  """Docstring."""
  graph = LogicalGraph(nodes=[LogicalNode(id="input", kind="Input")], edges=[])
  pass_ = SwiGLUFusionPass()
  new_graph = pass_.apply(graph)
  assert len(new_graph.nodes) == 1


def test_swiglu_defusion_pass():
  """Docstring."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="input", kind="Input"),
      LogicalNode(id="layer_swiglu", kind="SwiGLU"),
      LogicalNode(id="layer_out", kind="Other"),
      LogicalNode(id="unrelated1", kind="Other"),
      LogicalNode(id="unrelated2", kind="Other"),
    ],
    edges=[
      LogicalEdge(source="input", target="layer_swiglu"),
      LogicalEdge(source="layer_swiglu", target="layer_out"),
      LogicalEdge(source="unrelated1", target="unrelated2"),
    ],
  )
  pass_ = SwiGLUDefusionPass()
  new_graph = pass_.apply(graph)
  assert any(n.id == "layer_gate_proj" for n in new_graph.nodes)
  assert any(n.id == "layer_up_proj" for n in new_graph.nodes)
  assert any(e.source == "unrelated1" for e in new_graph.edges)


def test_swiglu_defusion_pass_no_match():
  """Docstring."""
  graph = LogicalGraph(nodes=[LogicalNode(id="input", kind="Input")], edges=[])
  pass_ = SwiGLUDefusionPass()
  new_graph = pass_.apply(graph)
  assert len(new_graph.nodes) == 1


def test_vision_patch_fusion_pass():
  """Docstring."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="input", kind="Input"),
      LogicalNode(id="patch_conv", kind="Conv2d"),
      LogicalNode(id="output", kind="Output"),
      LogicalNode(id="unrelated1", kind="Other"),
      LogicalNode(id="unrelated2", kind="Other"),
    ],
    edges=[
      LogicalEdge(source="input", target="patch_conv"),
      LogicalEdge(source="patch_conv", target="output"),
      LogicalEdge(source="unrelated1", target="unrelated2"),
    ],
  )
  pass_ = VisionPatchEmbeddingFusionPass()
  new_graph = pass_.apply(graph)
  assert any(n.kind == "VisionPatchEmbedding" for n in new_graph.nodes)
  assert any(e.source == "unrelated1" for e in new_graph.edges)


def test_vision_patch_fusion_pass_no_match():
  """Docstring."""
  graph = LogicalGraph(nodes=[LogicalNode(id="input", kind="Input")], edges=[])
  pass_ = VisionPatchEmbeddingFusionPass()
  new_graph = pass_.apply(graph)
  assert len(new_graph.nodes) == 1


def test_vision_patch_defusion_pass():
  """Docstring."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="input", kind="Input"),
      LogicalNode(id="patch_embed", kind="VisionPatchEmbedding"),
      LogicalNode(id="output", kind="Output"),
      LogicalNode(id="unrelated1", kind="Other"),
      LogicalNode(id="unrelated2", kind="Other"),
    ],
    edges=[
      LogicalEdge(source="input", target="patch_embed"),
      LogicalEdge(source="patch_embed", target="output"),
      LogicalEdge(source="unrelated1", target="unrelated2"),
    ],
  )
  pass_ = VisionPatchEmbeddingDefusionPass()
  new_graph = pass_.apply(graph)
  assert any(n.kind == "Conv2d" for n in new_graph.nodes)
  assert any(e.source == "unrelated1" for e in new_graph.edges)


def test_vision_patch_defusion_pass_no_match():
  """Docstring."""
  graph = LogicalGraph(nodes=[LogicalNode(id="input", kind="Input")], edges=[])
  pass_ = VisionPatchEmbeddingDefusionPass()
  new_graph = pass_.apply(graph)
  assert len(new_graph.nodes) == 1
