"""Test suite for the Qwen Fusion module."""

from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode
from ml_switcheroo.core.compiler.qwen_fusion import (
  SwiGLUDefusionPass,
  SwiGLUFusionPass,
  VisionPatchEmbeddingDefusionPass,
  VisionPatchEmbeddingFusionPass,
)


def test_swiglu_fusion_pass():
  """Verifies the behavior of swiglu fusion pass."""
  nodes = {
    "input": LogicalNode(id="input", op_type="Input"),
    "mlp_gate_proj": LogicalNode(id="mlp_gate_proj", op_type="Linear"),
    "mlp_up_proj": LogicalNode(id="mlp_up_proj", op_type="Linear"),
    "mlp_down_proj": LogicalNode(id="mlp_down_proj", op_type="Linear"),
    "other": LogicalNode(id="other", op_type="Other"),
  }
  graph = LogicalGraph(
    nodes=nodes,
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
  node_ids = set(fused_graph.nodes.keys())
  assert "mlp_swiglu" in node_ids
  assert "mlp_gate_proj" not in node_ids
  assert "mlp_up_proj" not in node_ids
  assert "other" in node_ids
  edges = [(e.source, e.target) for e in fused_graph.edges]
  assert ("input", "mlp_swiglu") in edges
  assert ("mlp_swiglu", "mlp_down_proj") in edges
  assert ("input", "other") in edges


def test_swiglu_fusion_pass_no_match():
  """Verifies the behavior of swiglu fusion pass no match."""
  graph = LogicalGraph(nodes={"mlp_gate_proj": LogicalNode(id="mlp_gate_proj", op_type="Linear")})
  pass_ = SwiGLUFusionPass()
  fused_graph = pass_.apply(graph)
  assert len(fused_graph.nodes) == 1


def test_swiglu_defusion_pass():
  """Verifies the behavior of swiglu defusion pass."""
  nodes = {
    "input": LogicalNode(id="input", op_type="Input"),
    "mlp_swiglu": LogicalNode(id="mlp_swiglu", op_type="SwiGLU"),
    "mlp_down_proj": LogicalNode(id="mlp_down_proj", op_type="Linear"),
    "other": LogicalNode(id="other", op_type="Other"),
  }
  graph = LogicalGraph(
    nodes=nodes,
    edges=[LogicalEdge("input", "mlp_swiglu"), LogicalEdge("mlp_swiglu", "mlp_down_proj"), LogicalEdge("input", "other")],
  )
  pass_ = SwiGLUDefusionPass()
  defused_graph = pass_.apply(graph)
  node_ids = set(defused_graph.nodes.keys())
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
  """Verifies the behavior of vision patch fusion pass."""
  nodes = {
    "input": LogicalNode(id="input", op_type="Input"),
    "patch_conv": LogicalNode(id="patch_conv", op_type="Conv2d"),
    "flatten": LogicalNode(id="flatten", op_type="Flatten"),
    "other": LogicalNode(id="other", op_type="Other"),
  }
  graph = LogicalGraph(
    nodes=nodes,
    edges=[LogicalEdge("input", "patch_conv"), LogicalEdge("patch_conv", "flatten"), LogicalEdge("input", "other")],
  )
  pass_ = VisionPatchEmbeddingFusionPass()
  fused_graph = pass_.apply(graph)
  node_ids = set(fused_graph.nodes.keys())
  assert "patch_patch_embed" in node_ids
  assert "patch_conv" not in node_ids
  assert "other" in node_ids
  edges = [(e.source, e.target) for e in fused_graph.edges]
  assert ("input", "patch_patch_embed") in edges
  assert ("patch_patch_embed", "flatten") in edges
  assert ("input", "other") in edges


def test_vision_patch_defusion_pass():
  """Verifies the behavior of vision patch defusion pass."""
  nodes = {
    "input": LogicalNode(id="input", op_type="Input"),
    "patch_patch_embed": LogicalNode(id="patch_patch_embed", op_type="VisionPatchEmbedding"),
    "flatten": LogicalNode(id="flatten", op_type="Flatten"),
    "other": LogicalNode(id="other", op_type="Other"),
  }
  graph = LogicalGraph(
    nodes=nodes,
    edges=[
      LogicalEdge("input", "patch_patch_embed"),
      LogicalEdge("patch_patch_embed", "flatten"),
      LogicalEdge("input", "other"),
    ],
  )
  pass_ = VisionPatchEmbeddingDefusionPass()
  defused_graph = pass_.apply(graph)
  node_ids = set(defused_graph.nodes.keys())
  assert "patch_patch_embed" not in node_ids
  assert "patch_conv" in node_ids
  assert "other" in node_ids
  edges = [(e.source, e.target) for e in defused_graph.edges]
  assert ("input", "patch_conv") in edges
  assert ("patch_conv", "flatten") in edges
  assert ("input", "other") in edges


def test_swiglu_fusion_pass_no_match_prefix():
  """Verifies the behavior when prefixes don't match."""
  nodes = {
    "layer1_gate_proj": LogicalNode(id="layer1_gate_proj", op_type="Linear"),
    "layer2_up_proj": LogicalNode(id="layer2_up_proj", op_type="Linear"),
  }
  graph = LogicalGraph(
    nodes=nodes,
    edges=[],
  )
  pass_ = SwiGLUFusionPass()
  fused = pass_.apply(graph)
  assert len(fused.nodes) == 2


def test_swiglu_defusion_pass_no_match():
  """Verifies the behavior when no swiglu exists."""
  graph = LogicalGraph(nodes={"gate_proj": LogicalNode(id="gate_proj", op_type="Linear")})
  pass_ = SwiGLUDefusionPass()
  defused = pass_.apply(graph)
  assert len(defused.nodes) == 1


def test_vision_patch_fusion_no_match():
  """Verifies the behavior when no patch node exists."""
  graph = LogicalGraph(nodes={"conv1": LogicalNode(id="conv1", op_type="Conv2d")})
  pass_ = VisionPatchEmbeddingFusionPass()
  fused = pass_.apply(graph)
  assert len(fused.nodes) == 1


def test_vision_patch_defusion_no_match():
  """Verifies the behavior when no patch embed node exists."""
  graph = LogicalGraph(nodes={"conv1": LogicalNode(id="conv1", op_type="Conv2d")})
  pass_ = VisionPatchEmbeddingDefusionPass()
  defused = pass_.apply(graph)
  assert len(defused.nodes) == 1
