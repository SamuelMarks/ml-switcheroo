"""
SASS Instruction Analyzer.

This module implements heuristics to extract high-level semantic parameters
(like Loop Bounds, Kernel Sizes, etc.) by analyzing the raw SASS instructions
within a basic block.

It enables the "Lifting" process to recover parameters like `kernel_size=3`
purely by observing the loop constraints defined in the assembly, without
relying on metadata side-channels.
"""

from typing import List, Dict, Any, Optional
from ml_switcheroo.core.sass.nodes import Instruction, Immediate


class SassAnalyzer:
  """
  Analyzes sequences of SASS instructions to reverse-engineer high-level parameters.
  """

  @staticmethod
  def analyze_block(kind: str, instructions: List[Instruction]) -> Dict[str, Any]:
    """
    Extracts metadata from a block of instructions based on the operation kind.

    Args:
        kind (str): The operation type (e.g. "Conv2d", "Linear").
        instructions (List[Instruction]): The assembly lines inside the block.

    Returns:
        Dict[str, Any]: Extracted parameters (e.g., {"kernel_size": 3}).
    """
    metadata = {}

    # Heuristic 1: Loop Bounds
    # Look for ISETP.LT.AND (Integer Set Predicate Less-Than)
    # Usage: ISETP.LT.AND P0, PT, Reg, Immediate, PT;
    # The Immediate value typically represents the loop limit.
    loop_limits = []
    for inst in instructions:
      if inst.opcode == "ISETP.LT.AND":
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
      # We take the max found limit as the kernel_size estimate.
      # (Assuming square kernels for simplicity in extraction)
      k_size = max(loop_limits)
      metadata["kernel_size"] = k_size
      # Torch signature default is just the int if square
      metadata["arg_2"] = k_size  # Positional mapping helper

    elif kind == "Linear":
      # Linear usually has 1 inner dot-product loop over input features.
      # The limit is in_features.
      if loop_limits:
        feat_dim = max(loop_limits)
        metadata["in_features"] = feat_dim
        # For Linear(in, out), 'in' is usually arg 0
        metadata["arg_0"] = feat_dim

    return metadata
