"""
Auto-Sharding Inference Pass.

This module implements a compiler pass that analyzes an unannotated `LogicalGraph`
(e.g., ingested from Hugging Face PyTorch models) and infers distributed sharding constraints
(e.g., for MaxText/NNX targets) based on standard tensor-parallel and FSDP heuristics.
"""

from typing import Dict, Optional

from ml_switcheroo.compiler.ir import LogicalGraph, LogicalMesh, PartitionSpec


class ShardingInferencePass:
  """
  Analyzes a graph and injects LogicalMesh and PartitionSpec annotations.

  Heuristics:
  - Linear layers with 'q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj' -> Column Parallel (None, "tensor").
  - Linear layers with 'o_proj', 'down_proj' -> Row Parallel ("tensor", None).
  - Embedding layers -> Row Parallel ("tensor", None) along vocab dimension.
  """

  def __init__(self, mesh: Optional[LogicalMesh] = None):
    """
    Initialize the sharding pass.

    Args:
        mesh: Optional target mesh. If None, a default 1D data mesh is assumed.
    """
    self.mesh = mesh or LogicalMesh(shape={"data": 1, "tensor": 1})

  def apply(self, graph: LogicalGraph) -> LogicalGraph:
    """
    Mutates the graph by injecting sharding annotations.

    Args:
        graph: The LogicalGraph to annotate.

    Returns:
        The annotated LogicalGraph (mutated in-place, but returned for chaining).
    """
    graph.mesh = self.mesh

    for node in graph.nodes:
      # Apply to Conv as well for Vision Patch fallback
      if node.kind in ["Linear", "Embedding", "Conv3d", "Conv2d"]:
        name = node.id.lower()
        if any(x in name for x in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]):
          # Column Parallel: shard the output dimension
          node.sharding = PartitionSpec(axes=(None, "tensor"))
        elif any(x in name for x in ["o_proj", "down_proj", "embed"]):
          # Row Parallel: shard the input dimension
          node.sharding = PartitionSpec(axes=("tensor", None))
        else:
          # Default FSDP-like Data Parallel fallback
          node.sharding = PartitionSpec(axes=("data", None))

    return graph
