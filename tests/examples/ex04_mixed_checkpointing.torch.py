"""Example demonstrating mixed checkpointing in PyTorch.

This module provides a minimal test case illustrating a mix of standard PyTorch
operations (which are eligible for automatic conversion, e.g., to JAX) and
framework-specific checkpointing utilities (`torch.utils.checkpoint.checkpoint`).
The checkpointing call behaves as an 'Escape Hatch' in translation pipelines
because its custom execution/backward semantics are not directly mapped
within basic operator-level translation tables.
"""

import torch
import torch.utils.checkpoint as checkpoint


def explicit_graph_step(x):
  """Performs an explicit graph step using standard operations and checkpointing.

  This function computes the absolute value of the input tensor and then
  executes a checkpointed block that doubles the intermediate value.
  It is designed to verify the translation pipeline's handling of standard
  operations mixed with framework-specific, non-mapped checkpoint utilities.

  Args:
    x (torch.Tensor): The input tensor to be processed.

  Returns:
    torch.Tensor: The output tensor resulting from the checkpointed operation.
  """
  # This standard op SHOULD be converted to JAX
  val = torch.abs(x)

  # This framework-specific utility should trigger the Escape Hatch
  # because it is not mapped in the semantics and requires explicit handling.
  out = checkpoint.checkpoint(lambda v: v * 2, val)

  return out
