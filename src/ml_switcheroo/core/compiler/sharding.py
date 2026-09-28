"""Auto-Sharding Inference Pass.

This module implements a compiler pass that analyzes an unannotated `LogicalGraph`
(e.g., ingested from Hugging Face PyTorch models) and infers distributed sharding constraints
(e.g., for JAX/NNX targets) based on standard tensor-parallel and FSDP heuristics.
"""

from typing import Any, Optional, Sequence

from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalMesh, PartitionSpec


def estimate_tensor_memory(
  shape: Sequence[Any],
  dtype: str = "float32",
) -> float:
  """Estimate tensor memory in bytes using analytical memory models.

  Args:
      shape: Tensor dimensions sequence.
      dtype: Tensor datatype string (e.g. 'float32', 'int32').

  Returns:
      float: Estimated size in bytes.
  """
  try:
    from ml_ecosystem_snapshots.compliance import estimate_tensor_memory_bytes

    int_shape = [int(x) for x in shape]
    return float(estimate_tensor_memory_bytes(int_shape, dtype))
  except Exception:
    import math

    bits = 32
    if "64" in dtype:
      bits = 64
    elif "16" in dtype:
      bits = 16
    elif "8" in dtype:
      bits = 8
    elements = math.prod([int(x) for x in shape]) if shape else 0
    return float(elements * (bits // 8))


class ShardingInferencePass:
  """Analyze a graph and injects LogicalMesh and PartitionSpec annotations.

  Heuristics:
  - Linear layers with 'q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj' -> Column Parallel (None, "tensor").
  - Linear layers with 'o_proj', 'down_proj' -> Row Parallel ("tensor", None).
  - Embedding layers -> Row Parallel ("tensor", None) along vocab dimension.
  """

  def __init__(self, mesh: Optional[LogicalMesh] = None):
    """Initialize the sharding pass.

    Args:
        mesh: Optional target mesh. If None, a default 1D data mesh is assumed.

    """
    self.mesh = mesh or LogicalMesh(shape={"data": 1, "tensor": 1})

  def estimate_graph_memory_bytes(self, graph: LogicalGraph) -> float:
    """Estimate total parameter and activation tensor memory in bytes for the graph.

    Args:
        graph: The LogicalGraph to evaluate.

    Returns:
        float: Total estimated memory in bytes.
    """
    total = 0.0
    for node in graph.nodes.values():
      shape = getattr(node, "shape", None) or node.attributes.get("shape")
      dtype = getattr(node, "dtype", None) or node.attributes.get("dtype", "float32")
      if isinstance(shape, (list, tuple)):
        total += estimate_tensor_memory(shape, str(dtype))
    return total

  def apply(self, graph: LogicalGraph) -> LogicalGraph:
    """Mutate the graph by injecting sharding annotations.

    Args:
        graph: The LogicalGraph to annotate.

    Returns:
        The annotated LogicalGraph (mutated in-place, but returned for chaining).

    """
    graph.mesh = self.mesh

    for node in graph.nodes.values():
      op_type = getattr(node, "op_type", None) or getattr(node, "kind", "")
      # Apply to Conv as well for Vision Patch fallback
      if op_type in ["Linear", "Embedding", "Conv3d", "Conv2d"]:
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
