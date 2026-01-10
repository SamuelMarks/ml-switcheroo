"""
RDNA Instruction Analyzer.

This module implements heuristics to extract high-level semantic parameters
(like Loop Bounds, Kernel Sizes) by analyzing the raw RDNA instructions
within a basic block.

It enables the "Lifting" process to recover parameters like `kernel_size=3`
from `s_cmp_lt_i32 sX, 3` instructions inside a loop structure.
"""

from typing import Any, Dict, List

from ml_switcheroo.compiler.frontends.rdna.nodes import Immediate, Instruction


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
    loop_limits = []
    for inst in instructions:
      if inst.opcode == "s_cmp_lt_i32":
        for op in inst.operands:
          if isinstance(op, Immediate):
            loop_limits.append(op.value)

    if not loop_limits:
      return metadata

    if kind == "Conv2d":
      k_size = max(loop_limits)
      metadata["k"] = k_size
      metadata["arg_2"] = k_size

    elif kind == "Linear":
      if loop_limits:
        feat_dim = max(loop_limits)
        metadata["in_features"] = feat_dim
        metadata["arg_0"] = feat_dim

    return metadata
