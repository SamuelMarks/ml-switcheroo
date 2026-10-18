"""Test suite for the Pytorch module."""

import torch
import torch.nn as nn


def gelu_activation(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
  """Helper to gelu activation.

  Args:
      x: Test argument or fixture.
      approximate: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return nn.functional.gelu(x, approximate=approximate)
