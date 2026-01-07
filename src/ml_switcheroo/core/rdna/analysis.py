"""
RDNA Instruction Analyzer.

This module implements heuristics to extract high-level semantic parameters
(like Loop Bounds, Kernel Sizes) by analyzing the raw RDNA instructions
within a basic block.

It enables the "Lifting" process to recover parameters like `kernel_size=3`
from `s_cmp_lt_i32 sX, 3` instructions inside a loop structure.
"""

from typing import Any, Dict, List

from ml_switcheroo.core.rdna.nodes import Immediate, Instruction


class RdnaAnalyzer:
  """
  Analyzes sequences of RDNA instructions to reverse-engineer high-level parameters.
  """

  @staticmethod
  def analyze_block(kind: str, instructions: List[Instruction]) -> Dict[str, Any]:
    """
    Extracts metadata from a block of instructions based on the operation kind.

    Args:
        kind (str): The operation type (e.g. "Conv2d", "Linear").
        instructions (List[Instruction]): The assembly lines inside the block.

    Returns:
        Dict[str, Any]: Extracted parameters (e.g., {"k": 3}).
    """
    metadata = {}

    # Heuristic: Loop Bounds via Scalar Comparison
    # Look for s_cmp_lt_i32 (Scalar Compare Less Than Integer)
    # Usage: s_cmp_lt_i32 sX, Immediate
    # The Immediate value typically represents the loop limit.
    loop_limits = []
    for inst in instructions:
      if inst.opcode == "s_cmp_lt_i32":
        # Check operands for Immediate values
        for op in inst.operands:
          if isinstance(op, Immediate):
            loop_limits.append(op.value)

    if not loop_limits:
      return metadata

    # Apply Model-Specific Logic
    if kind == "Conv2d":
      # Conv2d usually has 2 spatial loops (Ky, Kx).
      # If we see limits like [3, 3], it's likely a 3x3 kernel.
      # We take the max found limit.
      k_size = max(loop_limits)
      metadata["k"] = k_size
      # Torch signature positional helper
      metadata["arg_2"] = k_size

    elif kind == "Linear":
      # Linear usually has 1 inner dot-product loop over input features.
      # The limit is in_features.
      if loop_limits:
        feat_dim = max(loop_limits)
        metadata["in_features"] = feat_dim
        # Linear(in, out) -> arg_0
        metadata["arg_0"] = feat_dim

    return metadata
