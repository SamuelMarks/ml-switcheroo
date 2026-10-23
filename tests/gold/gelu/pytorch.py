"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


def gelu_activation(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
  """Helper to gelu activation.

  Args:
      x: Primary input tensor or array operand.
      approximate: The approximate (str) for gelu activation.

  Returns:
      Resulting `torch.Tensor` from gelu activation.
  """
  return nn.functional.gelu(x, approximate=approximate)
