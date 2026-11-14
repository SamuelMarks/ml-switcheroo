"""RDNA RdnaInstruction Analyzer.

This module implements heuristics to extract high-level semantic parameters
(like Loop Bounds, Kernel Sizes) by analyzing the raw RDNA instructions
within a basic block.

It enables the "Lifting" process to recover parameters like `kernel_size=3`
from `s_cmp_lt_i32 sX, 3` instructions inside a loop structure.
"""

import typing

from ml_switcheroo.core.compiler.frontends.rdna.cst import RdnaImmediate
from ml_switcheroo.core.compiler.isa.analysis import IsaAnalyzer


class RdnaAnalyzer(IsaAnalyzer):
  """Analyze sequences of RDNA instructions to reverse-engineer high-level parameters.

  Attributes:
      CMP_OPCODE: The cmp opcode (`str`).
      IMMEDIATE_TYPE: The immediate type (`type[RdnaImmediate]`).
      LOOP_PARAMETER_RULES: The loop parameter rules (`typing.Mapping[str, tuple[str, str]]`).
  """

  CMP_OPCODE: str = "s_cmp_lt_i32"
  IMMEDIATE_TYPE: type[RdnaImmediate] = RdnaImmediate
  LOOP_PARAMETER_RULES: typing.Mapping[str, tuple[str, str]] = {
    "Conv1d": ("kernel_size", "arg_2"),
    "Conv2d": ("k", "arg_2"),
    "Conv3d": ("kernel_size", "arg_2"),
    "DepthwiseConv2d": ("kernel_size", "arg_2"),
    "ConvTranspose1d": ("kernel_size", "arg_2"),
    "ConvTranspose2d": ("kernel_size", "arg_2"),
    "ConvTranspose3d": ("kernel_size", "arg_2"),
    "AvgPool1d": ("kernel_size", "arg_2"),
    "MaxPool1d": ("kernel_size", "arg_2"),
    "AvgPool2d": ("kernel_size", "arg_2"),
    "MaxPool2d": ("kernel_size", "arg_2"),
    "AvgPool3d": ("kernel_size", "arg_2"),
    "MaxPool3d": ("kernel_size", "arg_2"),
    "AdaptiveAvgPool2d": ("kernel_size", "arg_2"),
    "AdaptiveMaxPool2d": ("kernel_size", "arg_2"),
    "Linear": ("in_features", "arg_0"),
    "Mean": ("elements", "arg_0"),
    "Sum": ("elements", "arg_0"),
    "MSELoss": ("elements", "arg_0"),
    "CrossEntropyLoss": ("elements", "arg_0"),
  }
