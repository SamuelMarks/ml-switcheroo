"""Graph Optimization Passes for Qwen3 and Qwen3-VL architectures.

Provides passes to:
1. Fuse separate gate and up projections into a single SwiGLU operation.
2. De-fuse SwiGLU back into separate gate and up projections.
3. Handle VisionPatchEmbedding conversions.
"""

from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode, PartitionSpec


class SwiGLUFusionPass:
  """Fuse separate gate_proj and up_proj nodes into a single SwiGLU node.

  Matches standard JAX/Flax Bonsai idioms.
  """

  def apply(self, graph: LogicalGraph) -> LogicalGraph:
    """Mutate graph to fuse SwiGLU.

    Args:
        graph: The logical graph to mutate.

    Returns:
        The mutated graph.
    """
    gate_nodes = {
      n.id: n
      for n in graph.nodes.values()
      if (getattr(n, "op_type", None) or getattr(n, "kind", "")) == "Linear" and "gate_proj" in n.id.lower()
    }
    up_nodes = {
      n.id: n
      for n in graph.nodes.values()
      if (getattr(n, "op_type", None) or getattr(n, "kind", "")) == "Linear" and "up_proj" in n.id.lower()
    }

    if not (gate_nodes and up_nodes):
      return graph

    for gate_id, gate_node in list(gate_nodes.items()):
      prefix = gate_id.replace("gate_proj", "")
      up_id = prefix + "up_proj"

      if up_id in up_nodes:
        # Snapshot current edges before mutating nodes
        current_edges = list(graph.edges)

        # Create fused node
        fused_id = prefix + "swiglu"
        fused_node = LogicalNode(
          id=fused_id,
          op_type="SwiGLU",
          attributes={"fused": "True", "original_gate": gate_id, "original_up": up_id},
          sharding=PartitionSpec(axes=(None, "tensor")),
        )

        # Replace nodes
        new_nodes = {n.id: n for n in graph.nodes.values() if n.id not in (gate_id, up_id)}
        new_nodes[fused_id] = fused_node

        # Replace edges
        new_edges = []
        for e in current_edges:
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

        for node in new_nodes.values():
          node.inputs = []
        return LogicalGraph(name=graph.name, nodes=new_nodes, edges=new_edges, mesh=graph.mesh)

    return graph


class SwiGLUDefusionPass:
  """Split a SwiGLU node into separate gate_proj and up_proj nodes."""

  def apply(self, graph: LogicalGraph) -> LogicalGraph:
    """Mutate graph to de-fuse SwiGLU.

    Args:
        graph: The logical graph to mutate.

    Returns:
        The mutated graph.
    """
    swiglu_nodes = {
      n.id: n for n in graph.nodes.values() if (getattr(n, "op_type", None) or getattr(n, "kind", "")) == "SwiGLU"
    }

    for fused_id, fused_node in list(swiglu_nodes.items()):
      # Snapshot current edges before mutating nodes
      current_edges = list(graph.edges)

      prefix = fused_id.replace("swiglu", "")
      gate_id = prefix + "gate_proj"
      up_id = prefix + "up_proj"

      gate_node = LogicalNode(id=gate_id, op_type="Linear", sharding=PartitionSpec(axes=(None, "tensor")))
      up_node = LogicalNode(id=up_id, op_type="Linear", sharding=PartitionSpec(axes=(None, "tensor")))

      new_nodes = {n.id: n for n in graph.nodes.values() if n.id != fused_id}
      new_nodes[gate_id] = gate_node
      new_nodes[up_id] = up_node

      new_edges = []
      for e in current_edges:
        if e.target == fused_id:
          new_edges.extend([LogicalEdge(source=e.source, target=gate_id), LogicalEdge(source=e.source, target=up_id)])
        elif e.source == fused_id:
          new_edges.extend([LogicalEdge(source=gate_id, target=e.target), LogicalEdge(source=up_id, target=e.target)])
        else:
          new_edges.append(e)

      for node in new_nodes.values():
        node.inputs = []
      return LogicalGraph(name=graph.name, nodes=new_nodes, edges=new_edges, mesh=graph.mesh)

    return graph


class VisionPatchEmbeddingFusionPass:
  """Elevate Conv2d patch layers to native VisionPatchEmbedding multi-modal ops."""

  def apply(self, graph: LogicalGraph) -> LogicalGraph:
    """Mutate graph to elevate VisionPatchEmbedding.

    Args:
        graph: The logical graph to mutate.

    Returns:
        The mutated graph.
    """
    conv_nodes = {
      n.id: n
      for n in graph.nodes.values()
      if (getattr(n, "op_type", None) or getattr(n, "kind", "")) == "Conv2d" and "patch" in n.id.lower()
    }

    for conv_id, conv_node in list(conv_nodes.items()):
      # Snapshot current edges before mutating nodes
      current_edges = list(graph.edges)

      fused_id = conv_id.replace("conv", "patch_embed").replace("patch_embed_2d", "patch_embed")
      node_attrs = getattr(conv_node, "attributes", {})
      fused_node = LogicalNode(
        id=fused_id,
        op_type="VisionPatchEmbedding",
        attributes=node_attrs.copy(),
        sharding=PartitionSpec(axes=("data", None, None, None)),
      )

      new_nodes = {n.id: n for n in graph.nodes.values() if n.id != conv_id}
      new_nodes[fused_id] = fused_node

      new_edges = []
      for e in current_edges:
        if e.target == conv_id:
          new_edges.append(LogicalEdge(source=e.source, target=fused_id))
        elif e.source == conv_id:
          new_edges.append(LogicalEdge(source=fused_id, target=e.target))
        else:
          new_edges.append(e)

      for node in new_nodes.values():
        node.inputs = []
      return LogicalGraph(name=graph.name, nodes=new_nodes, edges=new_edges, mesh=graph.mesh)

    return graph


class VisionPatchEmbeddingDefusionPass:
  """Lower VisionPatchEmbedding back to structural Conv2d equivalents."""

  def apply(self, graph: LogicalGraph) -> LogicalGraph:
    """Mutate graph to defuse VisionPatchEmbedding.

    Args:
        graph: The logical graph to mutate.

    Returns:
        The mutated graph.
    """
    patch_nodes = {
      n.id: n
      for n in graph.nodes.values()
      if (getattr(n, "op_type", None) or getattr(n, "kind", "")) == "VisionPatchEmbedding"
    }

    for patch_id, patch_node in list(patch_nodes.items()):
      # Snapshot current edges before mutating nodes
      current_edges = list(graph.edges)

      conv_id = patch_id.replace("patch_embed", "conv")
      node_attrs = getattr(patch_node, "attributes", {})
      conv_node = LogicalNode(
        id=conv_id,
        op_type="Conv2d",
        attributes=node_attrs.copy(),
        sharding=PartitionSpec(axes=("data", None, None, None)),
      )

      new_nodes = {n.id: n for n in graph.nodes.values() if n.id != patch_id}
      new_nodes[conv_id] = conv_node

      new_edges = []
      for e in current_edges:
        if e.target == patch_id:
          new_edges.append(LogicalEdge(source=e.source, target=conv_id))
        elif e.source == patch_id:
          new_edges.append(LogicalEdge(source=conv_id, target=e.target))
        else:
          new_edges.append(e)

      for node in new_nodes.values():
        node.inputs = []
      return LogicalGraph(name=graph.name, nodes=new_nodes, edges=new_edges, mesh=graph.mesh)

    return graph
