"""RDNA RdnaInstruction Analyzer.

This module implements heuristics to extract high-level semantic parameters
(like Loop Bounds, Kernel Sizes) by analyzing the raw RDNA instructions
within a basic block.

It enables the "Lifting" process to recover parameters like `kernel_size=3`
from `s_cmp_lt_i32 sX, 3` instructions inside a loop structure.
"""

from typing import Mapping, Tuple, Type

from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaImmediate
from ml_switcheroo.core.compiler.isa.analysis import IsaAnalyzer


class RdnaAnalyzer(IsaAnalyzer):
  """Analyze sequences of RDNA instructions to reverse-engineer high-level parameters."""

  CMP_OPCODE: str = "s_cmp_lt_i32"
  IMMEDIATE_TYPE: Type[RdnaImmediate] = RdnaImmediate
  LOOP_PARAMETER_RULES: Mapping[str, Tuple[str, str]] = {
    "Conv2d": ("k", "arg_2"),
    "Linear": ("in_features", "arg_0"),
  }
