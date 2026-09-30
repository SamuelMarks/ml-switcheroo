"""Auto-Sharding Inference and Memory Modeling Pass.

This module implements compiler passes that analyze an unannotated `LogicalGraph`
(e.g., ingested from PyTorch models or ONNX graphs) and infer distributed sharding
constraints across multiple target runtimes:
- JAX / Flax NNX (via `LogicalMesh` and `PartitionSpec`).
- PyTorch DTensor / FSDP2 (via DTensor placement annotations).
- Apple MLX (via distributed array annotations and diagnostic warnings).

It also provides analytical memory estimation supporting dynamic batch sizes,
symbolic dimension tuples, activation buffers, gradient states, and device placement.
"""

import math
from typing import Any, Dict, List, Optional, Sequence

from ml_switcheroo.core.compiler.ir import LogicalGraph, LogicalMesh, PartitionSpec
from ml_switcheroo.utils.console import log_warning


def estimate_tensor_memory(
  shape: Sequence[Any],
  dtype: str = "float32",
  symbolic_map: Optional[Dict[str, int]] = None,
  default_symbolic_dim: int = 1,
) -> float:
  """Estimate tensor memory in bytes using analytical memory models.

  Supports dynamic batch sizes and symbolic dimension names by resolving them
  against an optional symbol map or falling back to a default dimension size.

  Args:
      shape: Tensor dimensions sequence, containing integers, strings, or None.
      dtype: Tensor datatype string (e.g. 'float32', 'int32', 'bfloat16').
      symbolic_map: Optional dictionary mapping symbolic dimension names to integer values.
      default_symbolic_dim: Default integer size for unknown symbolic dimensions (default: 1).

  Returns:
      float: Estimated size in bytes.
  """
  sym_map = symbolic_map or {}
  resolved_dims: List[int] = []

  for dim in shape:
    if dim is None:
      resolved_dims.append(default_symbolic_dim)
    elif isinstance(dim, int):
      resolved_dims.append(dim)
    elif isinstance(dim, str):
      clean_dim = dim.strip()
      if clean_dim in sym_map:
        resolved_dims.append(sym_map[clean_dim])
      else:
        try:
          parsed_int = int(clean_dim)
          resolved_dims.append(parsed_int if parsed_int > 0 else default_symbolic_dim)
        except ValueError:
          resolved_dims.append(default_symbolic_dim)
    else:
      try:
        resolved_dims.append(int(dim))
      except (ValueError, TypeError):
        resolved_dims.append(default_symbolic_dim)

  try:
    from ml_ecosystem_snapshots.compliance import estimate_tensor_memory_bytes

    return float(estimate_tensor_memory_bytes(resolved_dims, dtype))
  except Exception:
    bits = 32
    dt = dtype.lower()
    if "64" in dt:
      bits = 64
    elif "16" in dt or "bf16" in dt:
      bits = 16
    elif "8" in dt:
      bits = 8
    elif "4" in dt:
      bits = 4

    elements = math.prod(resolved_dims) if resolved_dims else 0
    return float(elements * (bits // 8 if bits >= 8 else 1))


class ShardingInferencePass:
  """Analyze a graph and inject LogicalMesh and PartitionSpec annotations.

  Heuristics:
  - Linear layers with 'q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj' -> Column Parallel (None, "tensor").
  - Linear layers with 'o_proj', 'down_proj' -> Row Parallel ("tensor", None).
  - Embedding layers -> Row Parallel ("tensor", None) along vocab dimension.
  - Convolution and other linear layers -> Data Parallel ("data", None) fallback.
  """

  def __init__(self, mesh: Optional[LogicalMesh] = None) -> None:
    """Initialize the sharding pass.

    Args:
        mesh: Optional target mesh. If None, a default 1D data mesh is assumed.
    """
    self.mesh = mesh or LogicalMesh(shape={"data": 1, "tensor": 1})

  def estimate_graph_memory_bytes(
    self,
    graph: LogicalGraph,
    include_activations: bool = True,
    include_gradients: bool = True,
    workspace_overhead_ratio: float = 0.1,
    symbolic_map: Optional[Dict[str, int]] = None,
  ) -> float:
    """Estimate total parameter, activation, gradient, and workspace memory in bytes.

    Args:
        graph: The LogicalGraph to evaluate.
        include_activations: Whether to estimate activation tensor memory overhead.
        include_gradients: Whether to account for gradient buffer memory (typically equal to parameter memory).
        workspace_overhead_ratio: Fractional overhead allocated for execution workspace buffers (default: 0.1).
        symbolic_map: Optional dictionary mapping symbolic shape dimension names to integer sizes.

    Returns:
        float: Total estimated memory in bytes.
    """
    param_memory = 0.0
    activation_memory = 0.0

    for node in graph.nodes.values():
      shape = getattr(node, "shape", None) or node.attributes.get("shape")
      dtype = getattr(node, "dtype", None) or node.attributes.get("dtype", "float32")

      if isinstance(shape, (list, tuple)):
        tensor_bytes = estimate_tensor_memory(
          shape=shape,
          dtype=str(dtype),
          symbolic_map=symbolic_map,
        )
        op_type = getattr(node, "op_type", "") or getattr(node, "kind", "")
        if op_type in ["Linear", "Embedding", "Conv2d", "Conv3d", "Parameter", "Constant"]:
          param_memory += tensor_bytes
        else:
          activation_memory += tensor_bytes

    total_grad = param_memory if include_gradients else 0.0
    total_act = activation_memory if include_activations else 0.0
    base_memory = param_memory + total_grad + total_act
    workspace_memory = base_memory * max(0.0, workspace_overhead_ratio)

    return float(base_memory + workspace_memory)

  def infer_device_placement(
    self,
    graph: LogicalGraph,
    device_capacity_bytes: float = 16.0 * 1024 * 1024 * 1024,
    num_devices: int = 1,
  ) -> Dict[str, int]:
    """Partition and assigns layers across accelerator devices based on memory capacity.

    Splits model layers across devices when accumulated layer memory exceeds
    the threshold capacity of an accelerator device.

    Args:
        graph: The LogicalGraph whose layers are to be partitioned.
        device_capacity_bytes: Maximum memory capacity per device in bytes (default: 16 GB).
        num_devices: Number of target accelerator devices available (default: 1).

    Returns:
        Dictionary mapping node IDs to assigned integer device indices (0 to num_devices - 1).
    """
    placements: Dict[str, int] = {}
    current_device = 0
    current_device_memory = 0.0

    for node_id, node in graph.nodes.items():
      shape = getattr(node, "shape", None) or node.attributes.get("shape")
      dtype = getattr(node, "dtype", None) or node.attributes.get("dtype", "float32")

      node_mem = 0.0
      if isinstance(shape, (list, tuple)):
        node_mem = estimate_tensor_memory(shape, str(dtype))

      if (current_device_memory + node_mem) > device_capacity_bytes and current_device < (num_devices - 1):
        current_device += 1
        current_device_memory = 0.0

      placements[node_id] = current_device
      current_device_memory += node_mem
      node.device = f"cuda:{current_device}"

    return placements

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
      if op_type in ["Linear", "Embedding", "Conv3d", "Conv2d"]:
        name = node.id.lower()
        if any(x in name for x in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]):
          node.sharding = PartitionSpec(axes=(None, "tensor"))
        elif any(x in name for x in ["o_proj", "down_proj", "embed"]):
          node.sharding = PartitionSpec(axes=("tensor", None))
        else:
          node.sharding = PartitionSpec(axes=("data", None))

    return graph


class PyTorchDTensorShardingPass:
  """Inject PyTorch DTensor / FSDP2 sharding annotations into a LogicalGraph.

  Translates standard model parallelism patterns into PyTorch DTensor placement specs:
  - Column parallel (e.g. q_proj, k_proj): Shard(dim=1)
  - Row parallel (e.g. o_proj, down_proj): Shard(dim=0)
  - FSDP fallback (e.g. general layers): FSDP Shard(dim=0)
  """

  def __init__(self, world_size: int = 8) -> None:
    """Initialize the PyTorch DTensor sharding pass.

    Args:
        world_size: Total distributed worker count (default: 8).
    """
    self.world_size = world_size

  def apply(self, graph: LogicalGraph) -> LogicalGraph:
    """Inject PyTorch DTensor placement attributes into graph nodes.

    Args:
        graph: The LogicalGraph to annotate.

    Returns:
        The updated LogicalGraph.
    """
    for node in graph.nodes.values():
      op_type = getattr(node, "op_type", None) or getattr(node, "kind", "")
      if op_type in ["Linear", "Embedding", "Conv2d", "Conv3d"]:
        name = node.id.lower()
        if any(x in name for x in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]):
          node.attributes["dtensor_placement"] = "Shard(dim=1)"
          node.attributes["dtensor_world_size"] = self.world_size
        elif any(x in name for x in ["o_proj", "down_proj", "embed"]):
          node.attributes["dtensor_placement"] = "Shard(dim=0)"
          node.attributes["dtensor_world_size"] = self.world_size
        else:
          node.attributes["dtensor_placement"] = "FSDP(dim=0)"
          node.attributes["dtensor_world_size"] = self.world_size

    return graph


class MlxDistributedShardingPass:
  """Inject Apple MLX distributed array annotations and surfaces diagnostics.

  Apple MLX leverages unified memory architecture across Apple Silicon chips.
  This pass annotates nodes with MLX distributed collective markers and surfaces
  warning diagnostics when non-contiguous layer sharding is inferred.
  """

  def __init__(self, num_nodes: int = 1) -> None:
    """Initialize the Apple MLX distributed pass.

    Args:
        num_nodes: Number of distributed Apple Silicon nodes (default: 1).
    """
    self.num_nodes = num_nodes
    self.diagnostics: List[str] = []

  def apply(self, graph: LogicalGraph) -> LogicalGraph:
    """Annotate graph nodes with MLX distributed markers and records diagnostics.

    Args:
        graph: The LogicalGraph to process.

    Returns:
        The processed LogicalGraph.
    """
    for node in graph.nodes.values():
      op_type = getattr(node, "op_type", None) or getattr(node, "kind", "")
      if op_type in ["Linear", "Embedding", "Conv2d", "Conv3d"]:
        node.attributes["mlx_distributed"] = True
        node.attributes["mlx_nodes"] = self.num_nodes
        msg = f"MLX distributed sharding active for node {node.id}: unified memory ring reduction enforced."
        self.diagnostics.append(msg)
        log_warning(msg)

    return graph
