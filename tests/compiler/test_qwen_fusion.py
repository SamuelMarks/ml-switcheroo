"""
Tests for Qwen-specific Graph Optimization Passes.
"""

from ml_switcheroo.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge
from ml_switcheroo.compiler.qwen_fusion import (
  SwiGLUFusionPass,
  SwiGLUDefusionPass,
  VisionPatchEmbeddingFusionPass,
  VisionPatchEmbeddingDefusionPass,
)


def test_swiglu_fusion_pass():
  """Tests fusing gate_proj and up_proj."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="input", kind="Input"),
      LogicalNode(id="mlp_gate_proj", kind="Linear"),
      LogicalNode(id="mlp_up_proj", kind="Linear"),
      LogicalNode(id="mlp_down_proj", kind="Linear"),
      LogicalNode(id="other", kind="Other"),
    ],
    edges=[
      LogicalEdge("input", "mlp_gate_proj"),
      LogicalEdge("input", "mlp_up_proj"),
      LogicalEdge("mlp_gate_proj", "mlp_down_proj"),
      LogicalEdge("mlp_up_proj", "mlp_down_proj"),
      LogicalEdge("input", "other"),
    ],
  )

  pass_ = SwiGLUFusionPass()
  fused_graph = pass_.apply(graph)

  node_ids = {n.id for n in fused_graph.nodes}
  assert "mlp_swiglu" in node_ids
  assert "mlp_gate_proj" not in node_ids
  assert "mlp_up_proj" not in node_ids
  assert "other" in node_ids

  edges = [(e.source, e.target) for e in fused_graph.edges]
  assert ("input", "mlp_swiglu") in edges
  assert ("mlp_swiglu", "mlp_down_proj") in edges
  assert ("input", "other") in edges


def test_swiglu_fusion_pass_no_match():
  """Tests fusion skips when gate or up are missing."""
  graph = LogicalGraph(nodes=[LogicalNode(id="mlp_gate_proj", kind="Linear")])
  pass_ = SwiGLUFusionPass()
  fused_graph = pass_.apply(graph)
  assert len(fused_graph.nodes) == 1


def test_swiglu_defusion_pass():
  """Tests splitting swiglu back into gate and up."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="input", kind="Input"),
      LogicalNode(id="mlp_swiglu", kind="SwiGLU"),
      LogicalNode(id="mlp_down_proj", kind="Linear"),
      LogicalNode(id="other", kind="Other"),
    ],
    edges=[
      LogicalEdge("input", "mlp_swiglu"),
      LogicalEdge("mlp_swiglu", "mlp_down_proj"),
      LogicalEdge("input", "other"),
    ],
  )

  pass_ = SwiGLUDefusionPass()
  defused_graph = pass_.apply(graph)

  node_ids = {n.id for n in defused_graph.nodes}
  assert "mlp_swiglu" not in node_ids
  assert "mlp_gate_proj" in node_ids
  assert "mlp_up_proj" in node_ids
  assert "other" in node_ids

  edges = [(e.source, e.target) for e in defused_graph.edges]
  assert ("input", "mlp_gate_proj") in edges
  assert ("input", "mlp_up_proj") in edges
  assert ("mlp_gate_proj", "mlp_down_proj") in edges
  assert ("mlp_up_proj", "mlp_down_proj") in edges
  assert ("input", "other") in edges


def test_vision_patch_fusion_pass():
  """Tests elevating conv patch to vision patch embedding."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="input", kind="Input"),
      LogicalNode(id="patch_conv", kind="Conv2d"),
      LogicalNode(id="flatten", kind="Flatten"),
      LogicalNode(id="other", kind="Other"),
    ],
    edges=[
      LogicalEdge("input", "patch_conv"),
      LogicalEdge("patch_conv", "flatten"),
      LogicalEdge("input", "other"),
    ],
  )

  pass_ = VisionPatchEmbeddingFusionPass()
  fused_graph = pass_.apply(graph)

  node_ids = {n.id for n in fused_graph.nodes}
  assert "patch_patch_embed" in node_ids
  assert "patch_conv" not in node_ids
  assert "other" in node_ids

  edges = [(e.source, e.target) for e in fused_graph.edges]
  assert ("input", "patch_patch_embed") in edges
  assert ("patch_patch_embed", "flatten") in edges
  assert ("input", "other") in edges


def test_vision_patch_defusion_pass():
  """Tests lowering vision patch embedding back to conv."""
  graph = LogicalGraph(
    nodes=[
      LogicalNode(id="input", kind="Input"),
      LogicalNode(id="patch_patch_embed", kind="VisionPatchEmbedding"),
      LogicalNode(id="flatten", kind="Flatten"),
      LogicalNode(id="other", kind="Other"),
    ],
    edges=[
      LogicalEdge("input", "patch_patch_embed"),
      LogicalEdge("patch_patch_embed", "flatten"),
      LogicalEdge("input", "other"),
    ],
  )

  pass_ = VisionPatchEmbeddingDefusionPass()
  defused_graph = pass_.apply(graph)

  node_ids = {n.id for n in defused_graph.nodes}
  assert "patch_patch_embed" not in node_ids
  assert "patch_conv" in node_ids
  assert "other" in node_ids

  edges = [(e.source, e.target) for e in defused_graph.edges]
  assert ("input", "patch_conv") in edges
  assert ("patch_conv", "flatten") in edges
  assert ("input", "other") in edges
