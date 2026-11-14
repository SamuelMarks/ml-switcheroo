"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


def gelu_activation(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
  """Execute `gelu` to support gelu activation in pytorch tests.

  Args:
      x: Primary input tensor or array operand.
      approximate: Input `approximate` parameter value (`str`).

  Returns:
      Resulting `torch.Tensor` from gelu activation.
  """
  return nn.functional.gelu(x, approximate=approximate)
