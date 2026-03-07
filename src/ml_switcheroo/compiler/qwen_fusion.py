"""
Graph Optimization Passes for Qwen3 and Qwen3-VL architectures.

Provides passes to:
1. Fuse separate gate and up projections into a single SwiGLU operation.
2. De-fuse SwiGLU back into separate gate and up projections.
3. Handle VisionPatchEmbedding conversions.
"""

from ml_switcheroo.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge, PartitionSpec


class SwiGLUFusionPass:
  """
  Fuses separate gate_proj and up_proj nodes into a single SwiGLU node.
  Matches standard JAX/Flax Bonsai idioms.
  """

  def apply(self, graph: LogicalGraph) -> LogicalGraph:
    """Mutates graph to fuse SwiGLU."""
    gate_nodes = {n.id: n for n in graph.nodes if n.kind == "Linear" and "gate_proj" in n.id.lower()}
    up_nodes = {n.id: n for n in graph.nodes if n.kind == "Linear" and "up_proj" in n.id.lower()}

    if not (gate_nodes and up_nodes):
      return graph

    for gate_id, gate_node in list(gate_nodes.items()):
      prefix = gate_id.replace("gate_proj", "")
      up_id = prefix + "up_proj"

      if up_id in up_nodes:
        # Create fused node
        fused_id = prefix + "swiglu"
        fused_node = LogicalNode(
          id=fused_id,
          kind="SwiGLU",
          metadata={"fused": "True", "original_gate": gate_id, "original_up": up_id},
          sharding=PartitionSpec(axes=(None, "tensor")),
        )

        # Replace nodes
        new_nodes = [n for n in graph.nodes if n.id not in (gate_id, up_id)]
        new_nodes.append(fused_node)
        graph.nodes = new_nodes

        # Replace edges
        new_edges = []
        for e in graph.edges:
          if e.target in (gate_id, up_id):
            new_edge = LogicalEdge(source=e.source, target=fused_id)
            if new_edge not in new_edges:
              new_edges.append(new_edge)
          elif e.source in (gate_id, up_id):
            new_edge = LogicalEdge(source=fused_id, target=e.target)
            if new_edge not in new_edges:
              new_edges.append(new_edge)
          else:
            new_edges.append(e)

        graph.edges = new_edges
        break  # One fusion per pass

    return graph


class SwiGLUDefusionPass:
  """
  Splits a SwiGLU node into separate gate_proj and up_proj nodes.
  """

  def apply(self, graph: LogicalGraph) -> LogicalGraph:
    """Mutates graph to de-fuse SwiGLU."""
    swiglu_nodes = {n.id: n for n in graph.nodes if n.kind == "SwiGLU"}

    for fused_id, fused_node in list(swiglu_nodes.items()):
      prefix = fused_id.replace("swiglu", "")
      gate_id = prefix + "gate_proj"
      up_id = prefix + "up_proj"

      gate_node = LogicalNode(id=gate_id, kind="Linear", sharding=PartitionSpec(axes=(None, "tensor")))
      up_node = LogicalNode(id=up_id, kind="Linear", sharding=PartitionSpec(axes=(None, "tensor")))

      new_nodes = [n for n in graph.nodes if n.id != fused_id]
      new_nodes.extend([gate_node, up_node])
      graph.nodes = new_nodes

      new_edges = []
      for e in graph.edges:
        if e.target == fused_id:
          new_edges.extend([LogicalEdge(source=e.source, target=gate_id), LogicalEdge(source=e.source, target=up_id)])
        elif e.source == fused_id:
          new_edges.extend([LogicalEdge(source=gate_id, target=e.target), LogicalEdge(source=up_id, target=e.target)])
        else:
          new_edges.append(e)

      graph.edges = new_edges
      break

    return graph


class VisionPatchEmbeddingFusionPass:
  """
  Elevates Conv2d patch layers to native VisionPatchEmbedding multi-modal ops.
  """

  def apply(self, graph: LogicalGraph) -> LogicalGraph:
    """Mutates graph to elevate VisionPatchEmbedding."""
    conv_nodes = {n.id: n for n in graph.nodes if n.kind == "Conv2d" and "patch" in n.id.lower()}

    for conv_id, conv_node in list(conv_nodes.items()):
      fused_id = conv_id.replace("conv", "patch_embed").replace("patch_embed_2d", "patch_embed")
      fused_node = LogicalNode(
        id=fused_id,
        kind="VisionPatchEmbedding",
        metadata=conv_node.metadata.copy(),
        sharding=PartitionSpec(axes=("data", None, None, None)),
      )

      new_nodes = [n for n in graph.nodes if n.id != conv_id]
      new_nodes.append(fused_node)
      graph.nodes = new_nodes

      new_edges = []
      for e in graph.edges:
        if e.target == conv_id:
          new_edges.append(LogicalEdge(source=e.source, target=fused_id))
        elif e.source == conv_id:
          new_edges.append(LogicalEdge(source=fused_id, target=e.target))
        else:
          new_edges.append(e)

      graph.edges = new_edges
      break

    return graph


class VisionPatchEmbeddingDefusionPass:
  """
  Lowers VisionPatchEmbedding back to structural Conv2d equivalents.
  """

  def apply(self, graph: LogicalGraph) -> LogicalGraph:
    """Mutates graph to defuse VisionPatchEmbedding."""
    patch_nodes = {n.id: n for n in graph.nodes if n.kind == "VisionPatchEmbedding"}

    for patch_id, patch_node in list(patch_nodes.items()):
      conv_id = patch_id.replace("patch_embed", "conv")
      conv_node = LogicalNode(
        id=conv_id,
        kind="Conv2d",
        metadata=patch_node.metadata.copy(),
        sharding=PartitionSpec(axes=("data", None, None, None)),
      )

      new_nodes = [n for n in graph.nodes if n.id != patch_id]
      new_nodes.append(conv_node)
      graph.nodes = new_nodes

      new_edges = []
      for e in graph.edges:
        if e.target == patch_id:
          new_edges.append(LogicalEdge(source=e.source, target=conv_id))
        elif e.source == patch_id:
          new_edges.append(LogicalEdge(source=conv_id, target=e.target))
        else:
          new_edges.append(e)

      graph.edges = new_edges
      break

    return graph
