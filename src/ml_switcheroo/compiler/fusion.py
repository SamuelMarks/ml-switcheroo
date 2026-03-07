"""
Graph Optimization Passes for specific architectures (e.g. MaxText vs HF).

Provides passes to:
1. Fuse Q, K, V projections into a single QKV projection.
2. De-fuse a QKV projection back into separate Q, K, V projections.
"""

from typing import Dict, List, Optional
from ml_switcheroo.compiler.ir import LogicalGraph, LogicalNode, LogicalEdge, PartitionSpec


class QKVFusionPass:
  """
  Fuses separate q_proj, k_proj, v_proj nodes into a single qkv_proj node.
  """

  def apply(self, graph: LogicalGraph) -> LogicalGraph:
    """Mutates graph to fuse QKV."""
    # Find all Linear nodes
    q_nodes = {n.id: n for n in graph.nodes if n.kind == "Linear" and "q_proj" in n.id.lower()}
    k_nodes = {n.id: n for n in graph.nodes if n.kind == "Linear" and "k_proj" in n.id.lower()}
    v_nodes = {n.id: n for n in graph.nodes if n.kind == "Linear" and "v_proj" in n.id.lower()}

    if not (q_nodes and k_nodes and v_nodes):
      return graph

    # For simplicity, if we find any matching Q, K, V, we fuse the first ones we find.
    # A robust implementation would group by source input.
    for q_id, q_node in q_nodes.items():
      # Try to find corresponding K and V with same prefix or structural group
      prefix = q_id.replace("q_proj", "")
      k_id = prefix + "k_proj"
      v_id = prefix + "v_proj"

      if k_id in k_nodes and v_id in v_nodes:
        k_node = k_nodes[k_id]
        v_node = v_nodes[v_id]

        # Create fused node
        fused_id = prefix + "qkv_proj"
        fused_node = LogicalNode(
          id=fused_id,
          kind="Linear",
          metadata={"fused": "True", "original_q": q_id, "original_k": k_id, "original_v": v_id},
          sharding=PartitionSpec(axes=(None, "tensor")),  # Column parallel
        )

        # Replace nodes
        new_nodes = [n for n in graph.nodes if n.id not in (q_id, k_id, v_id)]
        new_nodes.append(fused_node)
        graph.nodes = new_nodes

        # Replace edges
        new_edges = []
        for e in graph.edges:
          if e.target in (q_id, k_id, v_id):
            # Only add the edge once for the source
            new_edge = LogicalEdge(source=e.source, target=fused_id)
            if new_edge not in new_edges:
              new_edges.append(new_edge)
          elif e.source in (q_id, k_id, v_id):
            # Reroute output edges
            new_edges.append(LogicalEdge(source=fused_id, target=e.target))
          else:
            new_edges.append(e)

        graph.edges = new_edges
        break  # One fusion per pass for simplicity, could loop

    return graph


class QKVDefusionPass:
  """
  Splits a qkv_proj node into separate q_proj, k_proj, and v_proj nodes.
  """

  def apply(self, graph: LogicalGraph) -> LogicalGraph:
    """Mutates graph to de-fuse QKV."""
    qkv_nodes = {n.id: n for n in graph.nodes if n.kind == "Linear" and "qkv_proj" in n.id.lower()}

    for qkv_id, qkv_node in qkv_nodes.items():
      prefix = qkv_id.replace("qkv_proj", "")
      q_id = prefix + "q_proj"
      k_id = prefix + "k_proj"
      v_id = prefix + "v_proj"

      q_node = LogicalNode(id=q_id, kind="Linear", sharding=PartitionSpec(axes=(None, "tensor")))
      k_node = LogicalNode(id=k_id, kind="Linear", sharding=PartitionSpec(axes=(None, "tensor")))
      v_node = LogicalNode(id=v_id, kind="Linear", sharding=PartitionSpec(axes=(None, "tensor")))

      new_nodes = [n for n in graph.nodes if n.id != qkv_id]
      new_nodes.extend([q_node, k_node, v_node])
      graph.nodes = new_nodes

      new_edges = []
      for e in graph.edges:
        if e.target == qkv_id:
          new_edges.extend(
            [
              LogicalEdge(source=e.source, target=q_id),
              LogicalEdge(source=e.source, target=k_id),
              LogicalEdge(source=e.source, target=v_id),
            ]
          )
        elif e.source == qkv_id:
          # If output is used, we have to route from all three.
          # For a real implementation, subsequent logic expects Q, K, V.
          new_edges.extend(
            [
              LogicalEdge(source=q_id, target=e.target),
              LogicalEdge(source=k_id, target=e.target),
              LogicalEdge(source=v_id, target=e.target),
            ]
          )
        else:
          new_edges.append(e)

      graph.edges = new_edges
      break

    return graph
