"""Graph Optimization Passes for specific architectures (e.g., Transformer topologies).

Provides passes to:
1. Fuse Q, K, V projections into a single QKV projection.
2. De-fuse a QKV projection back into separate Q, K, V projections.
"""

from ml_switcheroo.core.compiler.ir import LogicalEdge, LogicalGraph, LogicalNode, PartitionSpec


class QKVFusionPass:
  """Fuse separate q_proj, k_proj, v_proj nodes into a single qkv_proj node."""

  def apply(self, graph: LogicalGraph) -> LogicalGraph:
    """Mutate graph to fuse QKV.

    Args:
        graph (LogicalGraph): The input logical graph.

    Returns:
        LogicalGraph: The mutated graph.
    """
    # Find all Linear nodes
    q_nodes = {
      n.id: n
      for n in graph.nodes.values()
      if (getattr(n, "op_type", None) or getattr(n, "kind", "")) == "Linear" and "q_proj" in n.id.lower()
    }
    k_nodes = {
      n.id: n
      for n in graph.nodes.values()
      if (getattr(n, "op_type", None) or getattr(n, "kind", "")) == "Linear" and "k_proj" in n.id.lower()
    }
    v_nodes = {
      n.id: n
      for n in graph.nodes.values()
      if (getattr(n, "op_type", None) or getattr(n, "kind", "")) == "Linear" and "v_proj" in n.id.lower()
    }

    if not (q_nodes and k_nodes and v_nodes):
      return graph

    for q_id, q_node in q_nodes.items():
      prefix = q_id.replace("q_proj", "")
      k_id = prefix + "k_proj"
      v_id = prefix + "v_proj"

      if k_id in k_nodes and v_id in v_nodes:
        # Snapshot current edges before mutating nodes
        current_edges = list(graph.edges)

        # Create fused node
        fused_id = prefix + "qkv_proj"
        fused_node = LogicalNode(
          id=fused_id,
          op_type="Linear",
          attributes={"fused": "True", "original_q": q_id, "original_k": k_id, "original_v": v_id},
          sharding=PartitionSpec(axes=(None, "tensor")),  # Column parallel
        )

        # Replace nodes
        new_nodes = {n.id: n for n in graph.nodes.values() if n.id not in (q_id, k_id, v_id)}
        new_nodes[fused_id] = fused_node

        # Replace edges
        new_edges = []
        for e in current_edges:
          if e.target in (q_id, k_id, v_id):
            new_edge = LogicalEdge(source=e.source, target=fused_id)
            if new_edge not in new_edges:
              new_edges.append(new_edge)
          elif e.source in (q_id, k_id, v_id):
            new_edge = LogicalEdge(source=fused_id, target=e.target)
            if new_edge not in new_edges:
              new_edges.append(new_edge)
          else:
            new_edges.append(e)

        for node in new_nodes.values():
          node.inputs = []
        return LogicalGraph(name=graph.name, nodes=new_nodes, edges=new_edges, mesh=graph.mesh)

    return graph


class QKVDefusionPass:
  """Split a qkv_proj node into separate q_proj, k_proj, and v_proj nodes."""

  def apply(self, graph: LogicalGraph) -> LogicalGraph:
    """Mutate graph to de-fuse QKV.

    Args:
        graph (LogicalGraph): The input logical graph.

    Returns:
        LogicalGraph: The mutated graph.
    """
    qkv_nodes = {
      n.id: n
      for n in graph.nodes.values()
      if (getattr(n, "op_type", None) or getattr(n, "kind", "")) == "Linear" and "qkv_proj" in n.id.lower()
    }

    for qkv_id, qkv_node in qkv_nodes.items():
      # Snapshot current edges before mutating nodes
      current_edges = list(graph.edges)

      prefix = qkv_id.replace("qkv_proj", "")
      q_id = prefix + "q_proj"
      k_id = prefix + "k_proj"
      v_id = prefix + "v_proj"

      q_node = LogicalNode(id=q_id, op_type="Linear", sharding=PartitionSpec(axes=(None, "tensor")))
      k_node = LogicalNode(id=k_id, op_type="Linear", sharding=PartitionSpec(axes=(None, "tensor")))
      v_node = LogicalNode(id=v_id, op_type="Linear", sharding=PartitionSpec(axes=(None, "tensor")))

      new_nodes = {n.id: n for n in graph.nodes.values() if n.id != qkv_id}
      new_nodes[q_id] = q_node
      new_nodes[k_id] = k_node
      new_nodes[v_id] = v_node

      new_edges = []
      for e in current_edges:
        if e.target == qkv_id:
          new_edges.extend(
            [
              LogicalEdge(source=e.source, target=q_id),
              LogicalEdge(source=e.source, target=k_id),
              LogicalEdge(source=e.source, target=v_id),
            ]
          )
        elif e.source == qkv_id:
          new_edges.extend(
            [
              LogicalEdge(source=q_id, target=e.target),
              LogicalEdge(source=k_id, target=e.target),
              LogicalEdge(source=v_id, target=e.target),
            ]
          )
        else:
          new_edges.append(e)

      for node in new_nodes.values():
        node.inputs = []
      return LogicalGraph(name=graph.name, nodes=new_nodes, edges=new_edges, mesh=graph.mesh)

    return graph
